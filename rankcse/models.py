import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

import transformers
from transformers import RobertaTokenizer
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel, RobertaModel, RobertaLMHead
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel, BertLMPredictionHead
from transformers.activations import gelu
from transformers.file_utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from transformers.modeling_outputs import SequenceClassifierOutput, BaseModelOutputWithPoolingAndCrossAttentions

class MLPLayer(nn.Module):
    """
    Head for getting sentence representations over RoBERTa/BERT's CLS representation.
    """

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = self.activation(x)

        return x

class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp

class Divergence(nn.Module):
    """
    Jensen-Shannon divergence, used to measure ranking consistency between similarity lists obtained from examples with two different dropout masks
    """
    def __init__(self, beta_):
        super(Divergence, self).__init__()
        self.kl = nn.KLDivLoss(reduction='batchmean', log_target=True)
        self.eps = 1e-7
        self.beta_ = beta_

    def forward(self, p: torch.tensor, q: torch.tensor):
        p, q = p.view(-1, p.size(-1)), q.view(-1, q.size(-1))
        m = (0.5 * (p + q)).log().clamp(min=self.eps)
        return self.beta_ * 0.5 * (self.kl(m, p.log()) + self.kl(m, q.log()))

class ListNet(nn.Module):
    """
    ListNet objective for ranking distillation; minimizes the cross entropy between permutation [top-1] probability distribution and ground truth obtained from teacher
    """
    def __init__(self, tau, gamma_):
        super(ListNet, self).__init__()
        self.teacher_temp_scaled_sim = Similarity(tau / 2)
        self.student_temp_scaled_sim = Similarity(tau)
        self.gamma_ = gamma_

    def forward(self, teacher_top1_sim_pred, student_top1_sim_pred):
        p = F.log_softmax(student_top1_sim_pred.fill_diagonal_(float('-inf')), dim=-1)
        q = F.softmax(teacher_top1_sim_pred.fill_diagonal_(float('-inf')), dim=-1)
        loss = -(q*p).nansum()  / q.nansum()
        return self.gamma_ * loss 

class ListMLE(nn.Module):
    """
    ListMLE objective for ranking distillation; maximizes the liklihood of the ground truth permutation (sorted indices of the ranking lists obtained from teacher) 
    """
    def __init__(self, tau, gamma_):
        super(ListMLE, self).__init__()
        self.temp_scaled_sim = Similarity(tau)
        self.gamma_ = gamma_ 
        self.eps = 1e-7

    def forward(self, teacher_top1_sim_pred, student_top1_sim_pred):

        y_pred = student_top1_sim_pred 
        y_true = teacher_top1_sim_pred

        # shuffle for randomised tie resolution
        random_indices = torch.randperm(y_pred.shape[-1])
        y_pred_shuffled = y_pred[:, random_indices]
        y_true_shuffled = y_true[:, random_indices]

        y_true_sorted, indices = y_true_shuffled.sort(descending=True, dim=-1)
        mask = y_true_sorted == -1
        preds_sorted_by_true = torch.gather(y_pred_shuffled, dim=1, index=indices)
        preds_sorted_by_true[mask] = float('-inf')
        max_pred_values, _ = preds_sorted_by_true.max(dim=1, keepdim=True)
        preds_sorted_by_true_minus_max = preds_sorted_by_true - max_pred_values
        cumsums = torch.cumsum(preds_sorted_by_true_minus_max.exp().flip(dims=[1]), dim=1).flip(dims=[1])
        observation_loss = torch.log(cumsums + self.eps) - preds_sorted_by_true_minus_max
        observation_loss[mask] = 0.0

        return self.gamma_ * torch.mean(torch.sum(observation_loss, dim=1))


class ChainTriangulationDistillation(nn.Module):
    """
    Cross-anchor triangulation ranking distillation.

    For each anchor i in the batch:
      View A: teacher ranks all other sentences relative to anchor i
      View C: teacher ranks all other sentences relative to a cross-anchor c
              (a mid-ranked sentence per teacher for anchor i)

    Both views rank the same candidate set. We reorder the student's
    similarities to match View A's coordinate space, compute pairwise
    difference matrices for both views, sum them, and apply the CoSENT
    all-pairs penalty:

        joint_diff = diff_A + diff_C   (aligned to same sentence ordering)
        loss = log(1 + Σ exp(λ · joint_diff[i,j]))   for i < j
    """
    def __init__(self, tau, gamma_, lambda_=1.0, top_k=32, ibn_lambda=None):
        super(ChainTriangulationDistillation, self).__init__()
        self.gamma_ = gamma_
        self.lambda_ = lambda_
        self.top_k = top_k
        self.ibn_lambda = ibn_lambda if ibn_lambda is not None else lambda_

    def forward(self, teacher_top1_sim_pred, student_top1_sim_pred):
        B = student_top1_sim_pred.size(0)
        device = student_top1_sim_pred.device
        K = min(self.top_k, B - 1)  # only compare top-K candidates per anchor

        # Mask diagonal (self-similarity) on teacher
        diag_mask = torch.eye(B, dtype=torch.bool, device=device)
        teacher_masked = teacher_top1_sim_pred.clone().masked_fill(diag_mask, float('-inf'))

        # --- View A: each sentence i is the anchor ---
        _, teacher_order_a = teacher_masked.sort(descending=True, dim=-1)
        # Only keep top-K candidates (like original's 16 positives)
        teacher_order_topk = teacher_order_a[:, :K]  # (B, K)
        student_sorted_a = torch.gather(student_top1_sim_pred, 1, teacher_order_topk)  # (B, K)

        # --- Cross-anchor: last in top-K (like original's furthest positive) ---
        cross_anchor_idx = teacher_order_topk[:, -1]  # (B,)

        # --- View C: rank from cross-anchor's perspective ---
        cross_student = student_top1_sim_pred[cross_anchor_idx]  # (B, B)
        # Align to View A's top-K ordering
        student_c_aligned = torch.gather(cross_student, 1, teacher_order_topk)  # (B, K)

        # --- Pairwise diffs and triangulation ---
        diff_a = student_sorted_a.unsqueeze(1) - student_sorted_a.unsqueeze(2)  # (B, K, K)
        diff_c = student_c_aligned.unsqueeze(1) - student_c_aligned.unsqueeze(2)  # (B, K, K)
        joint_diff = diff_a + diff_c

        # Upper-triangular mask (K x K, much smaller than B x B)
        triu_mask = torch.triu(torch.ones(K, K, device=device, dtype=torch.bool), diagonal=1)

        scaled = self.lambda_ * joint_diff
        scaled = scaled.masked_fill(~triu_mask, float('-inf'))
        scaled = scaled.masked_fill(torch.abs(joint_diff) < 1e-6, float('-inf'))
        scaled = torch.clamp(scaled, max=80.0)
        exp_terms = torch.exp(scaled)
        positions = torch.arange(K, device=device, dtype=exp_terms.dtype)
        pos_weight = 1.0 / (positions + 1.0)
        exp_terms = exp_terms * pos_weight.view(1, K, 1)
        ranked_loss = torch.log(1 + exp_terms.sum(dim=(1, 2))).mean()

        # --- IBN: top-K boundary enforcement ---
        # Worst candidate in top-K should beat all candidates outside top-K
        worst_topk_sim = student_sorted_a[:, -1]  # (B,) sim to K-th best
        teacher_order_outside = teacher_order_a[:, K:]  # (B, B-1-K) indices outside top-K
        student_outside = torch.gather(student_top1_sim_pred, 1, teacher_order_outside)  # (B, B-1-K)

        # diff = outside_sim - worst_topk_sim; penalize when outside > worst_topk
        ibn_diff = student_outside - worst_topk_sim.unsqueeze(1)  # (B, B-1-K)
        ibn_scaled = self.ibn_lambda * ibn_diff
        ibn_scaled = torch.clamp(ibn_scaled, max=80.0)
        ibn_loss = torch.log(1 + torch.exp(ibn_scaled).sum(dim=1)).mean()

        return self.gamma_ * (ranked_loss + ibn_loss)


class ChainTriangulationGreedyPartition(nn.Module):
    """
    Greedy list-partitioning variant of ChainTriangulationDistillation.

    Instead of per-anchor top-K from the full batch (each sentence plays many
    roles), this partitions the batch into mutually-exclusive lists of fixed
    size:

      Repeat num_lists times:
          - Pick the first available sentence as the list's anchor.
          - From the remaining available sentences, take the top-(list_size-1)
            ranked by teacher similarity to the anchor.
          - Mark all list_size sentences as used.

    For each list:
      - View A: anchor → its 15 ranked candidates.
      - View C: cross_anchor (the last candidate, weakest match) → same 15
        candidates in the same order.
      - CoSENT-style ranked loss on the joint pairwise differences.
      - Optional IBN: list's worst-ranked candidate must beat all sentences
        outside the list (in other lists).

    All operations within a list are independent of the surrounding batch, so
    the gradient signal per list is sharper than the diffuse top-K view used
    by the parent class.

    Args:
        tau: kept for signature compatibility with sibling distillation losses;
            not used by the loss (sort is invariant under positive scaling).
        gamma_: outer scale on the whole loss.
        lambda_: scale on the pairwise diffs inside the CoSENT log-sum-exp.
        list_size: number of sentences per list (1 anchor + list_size-1 cands).
        num_lists: how many lists to extract greedily from the batch.
        skip_last_n_lists: drop the K weakest lists (their leftover candidates
            are the least related). 0 = use all.
        ibn_lambda: scale on the IBN boundary loss. None = same as lambda_.
        use_ibn: include the cross-list IBN boundary term. Default True.
    """
    def __init__(self, tau, gamma_, lambda_=1.0, list_size=16, num_lists=8,
                 skip_last_n_lists=0, ibn_lambda=None, use_ibn=True):
        super().__init__()
        self.gamma_ = gamma_
        self.lambda_ = lambda_
        self.list_size = list_size
        self.num_lists = num_lists
        self.skip_last_n_lists = skip_last_n_lists
        self.ibn_lambda = ibn_lambda if ibn_lambda is not None else lambda_
        self.use_ibn = use_ibn

    @torch.no_grad()
    def _greedy_partition(self, teacher_sim):
        """Greedy partition into (num_lists, list_size) sentence indices."""
        B = teacher_sim.size(0)
        device = teacher_sim.device
        L, K = self.num_lists, self.list_size

        if L * K > B:
            raise ValueError(
                f"num_lists*list_size ({L}*{K}={L*K}) exceeds batch size {B}"
            )

        # Mask self-similarity so an anchor doesn't pick itself
        eye = torch.eye(B, dtype=torch.bool, device=device)
        ts = teacher_sim.masked_fill(eye, float('-inf'))

        available = torch.ones(B, dtype=torch.bool, device=device)
        lists = torch.zeros(L, K, dtype=torch.long, device=device)
        for l in range(L):
            # Pick anchor: first still-available sentence
            anchor_idx = torch.nonzero(available, as_tuple=False)[0].item()
            available[anchor_idx] = False
            lists[l, 0] = anchor_idx

            # Top-(K-1) most-similar still-available sentences for this anchor
            sims = ts[anchor_idx].clone()
            sims[~available] = float('-inf')
            _, top_idx = sims.topk(K - 1)
            available[top_idx] = False
            lists[l, 1:] = top_idx
        return lists  # (L, K), each row: [anchor, top1, top2, ..., top_{K-1}]

    def forward(self, teacher_top1_sim_pred, student_top1_sim_pred):
        B = student_top1_sim_pred.size(0)
        device = student_top1_sim_pred.device
        L, K = self.num_lists, self.list_size

        list_idx = self._greedy_partition(teacher_top1_sim_pred)  # (L, K)

        # ------------------------------------------------------------------
        # Build per-list View A and View C in student space.
        # ------------------------------------------------------------------
        anchor_ids = list_idx[:, 0]              # (L,)
        cand_ids = list_idx[:, 1:]               # (L, K-1)  candidates incl. cross-anchor
        cross_anchor_ids = list_idx[:, -1]       # (L,)  weakest candidate per list

        # View A: anchor_l's student sim to its 15 candidates
        student_a = student_top1_sim_pred[
            anchor_ids.unsqueeze(1).expand(-1, K - 1),
            cand_ids,
        ]  # (L, K-1)

        # View C: cross_anchor_l's student sim to the same 15 candidates
        # (note: candidate at position -1 is the cross_anchor itself → self-sim ≈ 1)
        student_c = student_top1_sim_pred[
            cross_anchor_ids.unsqueeze(1).expand(-1, K - 1),
            cand_ids,
        ]  # (L, K-1)

        # ------------------------------------------------------------------
        # Decide which lists actually contribute to the loss
        # ------------------------------------------------------------------
        L_eff = max(1, L - self.skip_last_n_lists)
        student_a = student_a[:L_eff]
        student_c = student_c[:L_eff]

        Kc = K - 1  # number of candidates per list (15 when K=16)

        # ------------------------------------------------------------------
        # Pairwise diffs → CoSENT-style joint penalty per list
        # ------------------------------------------------------------------
        # diff[l, i, j] = student[l, j] - student[l, i]   (sign matches existing
        # ChainTriangulationDistillation convention via broadcasting).
        diff_a = student_a.unsqueeze(1) - student_a.unsqueeze(2)   # (L_eff, Kc, Kc)
        diff_c = student_c.unsqueeze(1) - student_c.unsqueeze(2)
        joint_diff = diff_a + diff_c

        triu_mask = torch.triu(
            torch.ones(Kc, Kc, device=device, dtype=torch.bool), diagonal=1
        )

        scaled = self.lambda_ * joint_diff
        scaled = scaled.masked_fill(~triu_mask, float('-inf'))
        scaled = scaled.masked_fill(torch.abs(joint_diff) < 1e-6, float('-inf'))
        scaled = torch.clamp(scaled, max=80.0)
        exp_terms = torch.exp(scaled)

        # Position weighting (matches parent class behaviour): better-ranked
        # candidate (lower i) gets more weight.
        positions = torch.arange(Kc, device=device, dtype=exp_terms.dtype)
        pos_weight = 1.0 / (positions + 1.0)
        exp_terms = exp_terms * pos_weight.view(1, Kc, 1)

        ranked_loss = torch.log(1 + exp_terms.sum(dim=(1, 2))).mean()

        # ------------------------------------------------------------------
        # IBN: each list's weakest candidate must beat all OUT-OF-LIST sentences
        # ------------------------------------------------------------------
        ibn_loss = student_a.new_zeros(())
        if self.use_ibn:
            # Mask of sentences NOT in each list (L_eff, B)
            in_list = torch.zeros(L_eff, B, dtype=torch.bool, device=device)
            for l in range(L_eff):
                in_list[l, list_idx[l]] = True
            out_mask = ~in_list  # True where sentence is outside list l

            # Student sim from each list's anchor to all out-of-list sentences
            anchor_to_all = student_top1_sim_pred[anchor_ids[:L_eff]]  # (L_eff, B)
            # Worst-in-list = last candidate (cross_anchor) sim from anchor
            worst_in_list = student_a[:, -1].unsqueeze(1)  # (L_eff, 1)

            ibn_diff = anchor_to_all - worst_in_list           # (L_eff, B)
            ibn_diff = ibn_diff.masked_fill(~out_mask, float('-inf'))
            ibn_scaled = self.ibn_lambda * ibn_diff
            ibn_scaled = torch.clamp(ibn_scaled, max=80.0)
            ibn_loss = torch.log(1 + torch.exp(ibn_scaled).sum(dim=1)).mean()

        return self.gamma_ * (ranked_loss + ibn_loss)


class Pooler(nn.Module):
    """
    Parameter-free poolers to get the sentence embedding
    'cls': [CLS] representation with BERT/RoBERTa's MLP pooler.
    'cls_before_pooler': [CLS] representation without the original MLP pooler.
    'avg': average of the last layers' hidden states at each token.
    'avg_top2': average of the last two layers.
    'avg_first_last': average of the first and the last layers.
    """
    def __init__(self, pooler_type):
        super().__init__()
        self.pooler_type = pooler_type
        assert self.pooler_type in ["cls", "cls_before_pooler", "avg", "avg_top2", "avg_first_last"], "unrecognized pooling type %s" % self.pooler_type

    def forward(self, attention_mask, outputs):
        last_hidden = outputs.last_hidden_state
        pooler_output = outputs.pooler_output
        hidden_states = outputs.hidden_states

        if self.pooler_type in ['cls_before_pooler', 'cls']:
            return last_hidden[:, 0]
        elif self.pooler_type == "avg":
            return ((last_hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1))
        elif self.pooler_type == "avg_first_last":
            first_hidden = hidden_states[1]
            last_hidden = hidden_states[-1]
            pooled_result = ((first_hidden + last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        elif self.pooler_type == "avg_top2":
            second_last_hidden = hidden_states[-2]
            last_hidden = hidden_states[-1]
            pooled_result = ((last_hidden + second_last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        else:
            raise NotImplementedError


def cl_init(cls, config):
    """
    Contrastive learning class init function.
    """
    cls.pooler_type = cls.model_args.pooler_type
    cls.pooler = Pooler(cls.model_args.pooler_type)
    if cls.model_args.pooler_type == "cls":
        cls.mlp = MLPLayer(config)
    cls.sim = Similarity(temp=cls.model_args.temp)
    cls.div = Divergence(beta_=cls.model_args.beta_)
    if cls.model_args.distillation_loss == "listnet":
        cls.distillation_loss_fct = ListNet(cls.model_args.tau2, cls.model_args.gamma_)
    elif cls.model_args.distillation_loss == "listmle":
        cls.distillation_loss_fct = ListMLE(cls.model_args.tau2, cls.model_args.gamma_)
    elif cls.model_args.distillation_loss == "chain_triangulation":
        cls.distillation_loss_fct = ChainTriangulationDistillation(cls.model_args.tau2, cls.model_args.gamma_, cls.model_args.distillation_lambda)
    elif cls.model_args.distillation_loss == "chain_triangulation_greedy":
        cls.distillation_loss_fct = ChainTriangulationGreedyPartition(
            cls.model_args.tau2,
            cls.model_args.gamma_,
            cls.model_args.distillation_lambda,
            list_size=getattr(cls.model_args, "greedy_list_size", 16),
            num_lists=getattr(cls.model_args, "greedy_num_lists", 8),
            skip_last_n_lists=getattr(cls.model_args, "greedy_skip_last", 0),
        )
    else:
        raise NotImplementedError
    cls.init_weights()

def cl_forward(cls,
    encoder,
    input_ids=None,
    attention_mask=None,
    token_type_ids=None,
    position_ids=None,
    head_mask=None,
    inputs_embeds=None,
    labels=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
    mlm_input_ids=None,
    mlm_labels=None,
    teacher_top1_sim_pred=None,
):
    return_dict = return_dict if return_dict is not None else cls.config.use_return_dict
    ori_input_ids = input_ids
    batch_size = input_ids.size(0)
    # Number of sentences in one instance
    # 2: pair instance; 3: pair instance with a hard negative
    num_sent = input_ids.size(1)

    mlm_outputs = None
    # Flatten input for encoding
    input_ids = input_ids.view((-1, input_ids.size(-1))) # (bs * num_sent, len)
    attention_mask = attention_mask.view((-1, attention_mask.size(-1))) # (bs * num_sent len)
    if token_type_ids is not None:
        token_type_ids = token_type_ids.view((-1, token_type_ids.size(-1))) # (bs * num_sent, len)

    # Get raw embeddings
    outputs = encoder(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=True if cls.model_args.pooler_type in ['avg_top2', 'avg_first_last'] else False,
        return_dict=True,
    )

    # MLM auxiliary objective
    if mlm_input_ids is not None:
        mlm_input_ids = mlm_input_ids.view((-1, mlm_input_ids.size(-1)))
        mlm_outputs = encoder(
            mlm_input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True if cls.model_args.pooler_type in ['avg_top2', 'avg_first_last'] else False,
            return_dict=True,
        )

    # Pooling
    pooler_output = cls.pooler(attention_mask, outputs)
    pooler_output = pooler_output.view((batch_size, num_sent, pooler_output.size(-1))) # (bs, num_sent, hidden)

    # If using "cls", we add an extra MLP layer
    # (same as BERT's original implementation) over the representation.
    if cls.pooler_type == "cls":
        pooler_output = cls.mlp(pooler_output)

    # Separate representation
    z1, z2 = pooler_output[:,0], pooler_output[:,1]

    # Hard negative
    if num_sent == 3:
        z3 = pooler_output[:, 2]

    # Gather all embeddings if using distributed training
    if dist.is_initialized() and cls.training:
        # Gather hard negative
        if num_sent >= 3:
            z3_list = [torch.zeros_like(z3) for _ in range(dist.get_world_size())]
            dist.all_gather(tensor_list=z3_list, tensor=z3.contiguous())
            z3_list[dist.get_rank()] = z3
            z3 = torch.cat(z3_list, 0)

        # Dummy vectors for allgather
        z1_list = [torch.zeros_like(z1) for _ in range(dist.get_world_size())]
        z2_list = [torch.zeros_like(z2) for _ in range(dist.get_world_size())]
        # Allgather
        dist.all_gather(tensor_list=z1_list, tensor=z1.contiguous())
        dist.all_gather(tensor_list=z2_list, tensor=z2.contiguous())

        # Since allgather results do not have gradients, we replace the
        # current process's corresponding embeddings with original tensors
        z1_list[dist.get_rank()] = z1
        z2_list[dist.get_rank()] = z2
        # Get full batch embeddings: (bs x N, hidden)
        z1 = torch.cat(z1_list, 0)
        z2 = torch.cat(z2_list, 0)

    cos_sim = cls.sim(z1.unsqueeze(1), z2.unsqueeze(0))
    # Hard negative
    if num_sent >= 3:
        z1_z3_cos = cls.sim(z1.unsqueeze(1), z3.unsqueeze(0))
        cos_sim = torch.cat([cos_sim, z1_z3_cos], 1)

    labels = torch.arange(cos_sim.size(0)).long().to(cls.device)
    loss_fct = nn.CrossEntropyLoss()

    # Calculate loss with hard negatives
    if num_sent == 3:
        # Note that weights are actually logits of weights
        z3_weight = cls.model_args.hard_negative_weight
        weights = torch.tensor(
            [[0.0] * (cos_sim.size(-1) - z1_z3_cos.size(-1)) + [0.0] * i + [z3_weight] + [0.0] * (z1_z3_cos.size(-1) - i - 1) for i in range(z1_z3_cos.size(-1))]
        ).to(cls.device)
        cos_sim = cos_sim + weights

    loss = loss_fct(cos_sim, labels)

    # RankCSE - knowledge distillation loss 
    student_top1_sim_pred = cos_sim.clone()
    kd_loss = cls.distillation_loss_fct(teacher_top1_sim_pred.to(cls.device), student_top1_sim_pred)

    # RankCSE - self-distillation loss
    z1_z2_cos = cos_sim.clone()
    z2_z1_cos = cls.sim(z2.unsqueeze(1), z1.unsqueeze(0))
    sd_loss = cls.div(z1_z2_cos.softmax(dim=-1).clamp(min=1e-7), z2_z1_cos.softmax(dim=-1).clamp(min=1e-7))

    # L = L_infoNCE + L_consistency + L_distillation
    loss = loss + sd_loss + kd_loss

    # Calculate loss for MLM
    if mlm_outputs is not None and mlm_labels is not None:
        mlm_labels = mlm_labels.view(-1, mlm_labels.size(-1))
        prediction_scores = cls.lm_head(mlm_outputs.last_hidden_state)
        masked_lm_loss = loss_fct(prediction_scores.view(-1, cls.config.vocab_size), mlm_labels.view(-1))
        loss = loss + cls.model_args.mlm_weight * masked_lm_loss

    if not return_dict:
        output = (cos_sim,) + outputs[2:]
        return ((loss,) + output) if loss is not None else output
    return SequenceClassifierOutput(
        loss=loss,
        logits=cos_sim,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )


def sentemb_forward(
    cls,
    encoder,
    input_ids=None,
    attention_mask=None,
    token_type_ids=None,
    position_ids=None,
    head_mask=None,
    inputs_embeds=None,
    labels=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
):

    return_dict = return_dict if return_dict is not None else cls.config.use_return_dict

    outputs = encoder(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=True if cls.pooler_type in ['avg_top2', 'avg_first_last'] else False,
        return_dict=True,
    )

    pooler_output = cls.pooler(attention_mask, outputs)
    if cls.pooler_type == "cls" and not cls.model_args.mlp_only_train:
        pooler_output = cls.mlp(pooler_output)

    if not return_dict:
        return (outputs[0], pooler_output) + outputs[2:]

    return BaseModelOutputWithPoolingAndCrossAttentions(
        pooler_output=pooler_output,
        last_hidden_state=outputs.last_hidden_state,
        hidden_states=outputs.hidden_states,
    )


class BertForCL(BertPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, *model_args, **model_kargs):
        super().__init__(config)
        self.model_args = model_kargs["model_args"]
        self.bert = BertModel(config, add_pooling_layer=False)

        if self.model_args.do_mlm:
            self.lm_head = BertLMPredictionHead(config)

        cl_init(self, config)

    def forward(self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        sent_emb=False,
        mlm_input_ids=None,
        mlm_labels=None,
        teacher_top1_sim_pred=None,
    ):
        if sent_emb:
            return sentemb_forward(self, self.bert,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        else:
            return cl_forward(self, self.bert,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                mlm_input_ids=mlm_input_ids,
                mlm_labels=mlm_labels,
                teacher_top1_sim_pred=teacher_top1_sim_pred,
            )



class RobertaForCL(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, *model_args, **model_kargs):
        super().__init__(config)
        self.model_args = model_kargs["model_args"]
        self.roberta = RobertaModel(config, add_pooling_layer=False)

        if self.model_args.do_mlm:
            self.lm_head = RobertaLMHead(config)

        cl_init(self, config)

    def forward(self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        sent_emb=False,
        mlm_input_ids=None,
        mlm_labels=None,
        teacher_top1_sim_pred=None,
    ):
        if sent_emb:
            return sentemb_forward(self, self.roberta,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        else:
            return cl_forward(self, self.roberta,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                mlm_input_ids=mlm_input_ids,
                mlm_labels=mlm_labels,
                teacher_top1_sim_pred=teacher_top1_sim_pred,
            )

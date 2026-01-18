import math
from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...integrations.deepspeed import is_deepspeed_zero3_enabled
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask, _prepare_4d_causal_attention_mask
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_nllb_moe import NllbMoeConfig
def route_tokens(self, router_logits: torch.Tensor, input_dtype: torch.dtype=torch.float32, padding_mask: Optional[torch.LongTensor]=None) -> Tuple:
    """
        Computes the `dispatch_mask` and the `dispatch_weights` for each experts. The masks are adapted to the expert
        capacity.
        """
    nb_tokens = router_logits.shape[0]
    router_probs = nn.functional.softmax(router_logits, dim=-1, dtype=self.dtype).to(input_dtype)
    top_1_expert_index = torch.argmax(router_probs, dim=-1)
    top_1_mask = torch.nn.functional.one_hot(top_1_expert_index, num_classes=self.num_experts)
    if self.second_expert_policy == 'sampling':
        gumbel = torch.distributions.gumbel.Gumbel(0, 1).rsample
        router_logits += gumbel(router_logits.shape).to(router_logits.device)
    logits_except_top_1 = router_logits.masked_fill(top_1_mask.bool(), float('-inf'))
    top_2_expert_index = torch.argmax(logits_except_top_1, dim=-1)
    top_2_mask = torch.nn.functional.one_hot(top_2_expert_index, num_classes=self.num_experts)
    if self.normalize_router_prob_before_dropping:
        top_1_max_probs, top_2_max_probs = self.normalize_router_probabilities(router_probs, top_1_mask, top_2_mask)
    if self.second_expert_policy == 'random':
        top_2_max_probs = (router_probs * top_2_mask).sum(dim=1)
        sampled = 2 * top_2_max_probs > torch.rand_like(top_2_max_probs.float())
        top_2_mask = top_2_mask * sampled.repeat(self.num_experts, 1).transpose(1, 0)
    if padding_mask is not None and (not self.router_ignore_padding_tokens):
        if len(padding_mask.shape) == 4:
            padding_mask = padding_mask[:, :, -1, :].reshape(-1)[-nb_tokens:]
        non_padding = ~padding_mask.bool()
        top_1_mask = top_1_mask * non_padding.unsqueeze(-1).to(top_1_mask.dtype)
        top_2_mask = top_2_mask * non_padding.unsqueeze(-1).to(top_1_mask.dtype)
    if self.batch_prioritized_routing:
        importance_scores = -1 * router_probs.max(dim=1)[0]
        sorted_top_1_mask = top_1_mask[importance_scores.argsort(dim=0)]
        sorted_cumsum1 = (torch.cumsum(sorted_top_1_mask, dim=0) - 1) * sorted_top_1_mask
        locations1 = sorted_cumsum1[importance_scores.argsort(dim=0).argsort(dim=0)]
        sorted_top_2_mask = top_2_mask[importance_scores.argsort(dim=0)]
        sorted_cumsum2 = (torch.cumsum(sorted_top_2_mask, dim=0) - 1) * sorted_top_2_mask
        locations2 = sorted_cumsum2[importance_scores.argsort(dim=0).argsort(dim=0)]
        locations2 += torch.sum(top_1_mask, dim=0, keepdim=True)
    else:
        locations1 = torch.cumsum(top_1_mask, dim=0) - 1
        locations2 = torch.cumsum(top_2_mask, dim=0) - 1
        locations2 += torch.sum(top_1_mask, dim=0, keepdim=True)
    if not self.training and self.moe_eval_capacity_token_fraction > 0:
        self.expert_capacity = math.ceil(self.moe_eval_capacity_token_fraction * nb_tokens)
    else:
        capacity = 2 * math.ceil(nb_tokens / self.num_experts)
        self.expert_capacity = capacity if self.expert_capacity is None else self.expert_capacity
    top_1_mask = top_1_mask * torch.lt(locations1, self.expert_capacity)
    top_2_mask = top_2_mask * torch.lt(locations2, self.expert_capacity)
    if not self.normalize_router_prob_before_dropping:
        top_1_max_probs, top_2_max_probs = self.normalize_router_probabilities(router_probs, top_1_mask, top_2_mask)
    gates1 = top_1_max_probs[:, None] * top_1_mask
    gates2 = top_2_max_probs[:, None] * top_2_mask
    router_probs = gates1 + gates2
    return (top_1_mask, router_probs)
import math
from typing import List, Optional, Tuple, Union
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import ALL_LAYERNORM_LAYERS
from ...utils import (
from .configuration_mega import MegaConfig
def softmax_attention(self, query, key, padding_mask, causal_mask):
    """Standard softmax self-attention, as in the original Transformer paper"""
    seq_len = key.size(2)
    bias = self.rel_pos_bias(seq_len)
    if seq_len != query.size(2):
        if query.size(2) != 1:
            raise ValueError('Size mismatch between Q and K in softmax attention')
        bias = bias[-1:]
    query = query * self.scaling
    qk = torch.matmul(query, key.transpose(2, 3)) + bias
    if causal_mask is not None:
        additive_causal_mask = torch.zeros_like(causal_mask, dtype=qk.dtype)
        additive_causal_mask = additive_causal_mask.masked_fill((1 - causal_mask).bool(), float('-inf'))
        qk = qk + additive_causal_mask
    if padding_mask is not None:
        padding_mask = 1 - padding_mask
        padding_mask_all = padding_mask.all(dim=-1, keepdim=True)
        padding_mask = torch.logical_and(padding_mask, ~padding_mask_all)
        qk = qk.masked_fill(padding_mask.unsqueeze(2).to(torch.bool), float('-inf'))
    attn_weights = self.softmax(qk).type_as(qk)
    return attn_weights
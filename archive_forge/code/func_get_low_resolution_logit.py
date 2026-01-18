import math
from pathlib import Path
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch.utils.cpp_extension import load
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
from .configuration_mra import MraConfig
def get_low_resolution_logit(query, key, block_size, mask=None, value=None):
    """
    Compute low resolution approximation.
    """
    batch_size, seq_len, head_dim = query.size()
    num_block_per_row = seq_len // block_size
    value_hat = None
    if mask is not None:
        token_count = mask.reshape(batch_size, num_block_per_row, block_size).sum(dim=-1)
        query_hat = query.reshape(batch_size, num_block_per_row, block_size, head_dim).sum(dim=-2) / (token_count[:, :, None] + 1e-06)
        key_hat = key.reshape(batch_size, num_block_per_row, block_size, head_dim).sum(dim=-2) / (token_count[:, :, None] + 1e-06)
        if value is not None:
            value_hat = value.reshape(batch_size, num_block_per_row, block_size, head_dim).sum(dim=-2) / (token_count[:, :, None] + 1e-06)
    else:
        token_count = block_size * torch.ones(batch_size, num_block_per_row, dtype=torch.float, device=query.device)
        query_hat = query.reshape(batch_size, num_block_per_row, block_size, head_dim).mean(dim=-2)
        key_hat = key.reshape(batch_size, num_block_per_row, block_size, head_dim).mean(dim=-2)
        if value is not None:
            value_hat = value.reshape(batch_size, num_block_per_row, block_size, head_dim).mean(dim=-2)
    low_resolution_logit = torch.matmul(query_hat, key_hat.transpose(-1, -2)) / math.sqrt(head_dim)
    low_resolution_logit_row_max = low_resolution_logit.max(dim=-1, keepdims=True).values
    if mask is not None:
        low_resolution_logit = low_resolution_logit - 10000.0 * (token_count[:, None, :] * token_count[:, :, None] < 0.5).float()
    return (low_resolution_logit, token_count, low_resolution_logit_row_max, value_hat)
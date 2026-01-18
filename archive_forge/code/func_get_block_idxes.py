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
def get_block_idxes(low_resolution_logit, num_blocks, approx_mode, initial_prior_first_n_blocks, initial_prior_diagonal_n_blocks):
    """
    Compute the indices of the subset of components to be used in the approximation.
    """
    batch_size, total_blocks_per_row, _ = low_resolution_logit.shape
    if initial_prior_diagonal_n_blocks > 0:
        offset = initial_prior_diagonal_n_blocks // 2
        temp_mask = torch.ones(total_blocks_per_row, total_blocks_per_row, device=low_resolution_logit.device)
        diagonal_mask = torch.tril(torch.triu(temp_mask, diagonal=-offset), diagonal=offset)
        low_resolution_logit = low_resolution_logit + diagonal_mask[None, :, :] * 5000.0
    if initial_prior_first_n_blocks > 0:
        low_resolution_logit[:, :initial_prior_first_n_blocks, :] = low_resolution_logit[:, :initial_prior_first_n_blocks, :] + 5000.0
        low_resolution_logit[:, :, :initial_prior_first_n_blocks] = low_resolution_logit[:, :, :initial_prior_first_n_blocks] + 5000.0
    top_k_vals = torch.topk(low_resolution_logit.reshape(batch_size, -1), num_blocks, dim=-1, largest=True, sorted=False)
    indices = top_k_vals.indices
    if approx_mode == 'full':
        threshold = top_k_vals.values.min(dim=-1).values
        high_resolution_mask = (low_resolution_logit >= threshold[:, None, None]).float()
    elif approx_mode == 'sparse':
        high_resolution_mask = None
    else:
        raise ValueError(f'{approx_mode} is not a valid approx_model value.')
    return (indices, high_resolution_mask)
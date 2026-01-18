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
def sparse_mask(mask, indices, block_size=32):
    """
    Converts attention mask to a sparse mask for high resolution logits.
    """
    if len(mask.size()) != 2:
        raise ValueError('mask must be a 2-dimensional tensor.')
    if len(indices.size()) != 2:
        raise ValueError('indices must be a 2-dimensional tensor.')
    if mask.shape[0] != indices.shape[0]:
        raise ValueError('mask and indices must have the same size in the zero-th dimension.')
    batch_size, seq_len = mask.shape
    num_block = seq_len // block_size
    batch_idx = torch.arange(indices.size(0), dtype=torch.long, device=indices.device)
    mask = mask.reshape(batch_size, num_block, block_size)
    mask = mask[batch_idx[:, None], (indices % num_block).long(), :]
    return mask
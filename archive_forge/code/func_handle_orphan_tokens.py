import copy
import math
import warnings
from typing import Any, List, Optional, Tuple, Union
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import ALL_LAYERNORM_LAYERS, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
from .configuration_longt5 import LongT5Config
def handle_orphan_tokens(block_ids: torch.Tensor) -> torch.Tensor:
    block_ends = torch.arange(seq_len) % global_block_size == global_block_size - 1
    block_ends = block_ends.to(block_ids.device)
    true_block_ends = torch.logical_and(block_ends, block_ids >= 0)
    full_blocks = true_block_ends.sum(-1).unsqueeze(-1).type(block_ids.dtype) - 1
    block_ids = torch.where(block_ids < full_blocks, block_ids, full_blocks)
    return block_ids
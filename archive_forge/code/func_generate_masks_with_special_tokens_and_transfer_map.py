import copy
import math
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from ...activations import ACT2FN
from ...file_utils import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import meshgrid
from ...utils import is_accelerate_available, is_ninja_available, logging
from ...utils.backbone_utils import load_backbone
from ..auto import AutoModel
from .configuration_grounding_dino import GroundingDinoConfig
def generate_masks_with_special_tokens_and_transfer_map(input_ids: torch.LongTensor) -> Tuple[Tensor, Tensor]:
    """Generate attention mask between each pair of special tokens and positional ids.
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary.
    Returns:
        `tuple(torch.Tensor)` comprising attention mask between each special tokens and position_ids:
        - **attention_mask** (`torch.BoolTensor` of shape `(batch_size, sequence_length, sequence_length)`)
        - **position_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`)
    """
    batch_size, num_token = input_ids.shape
    special_tokens_mask = torch.zeros((batch_size, num_token), device=input_ids.device).bool()
    for special_token in SPECIAL_TOKENS:
        special_tokens_mask |= input_ids == special_token
    idxs = torch.nonzero(special_tokens_mask)
    attention_mask = torch.eye(num_token, device=input_ids.device).bool().unsqueeze(0).repeat(batch_size, 1, 1)
    position_ids = torch.zeros((batch_size, num_token), device=input_ids.device)
    previous_col = 0
    for i in range(idxs.shape[0]):
        row, col = idxs[i]
        if col == 0 or col == num_token - 1:
            attention_mask[row, col, col] = True
            position_ids[row, col] = 0
        else:
            attention_mask[row, previous_col + 1:col + 1, previous_col + 1:col + 1] = True
            position_ids[row, previous_col + 1:col + 1] = torch.arange(0, col - previous_col, device=input_ids.device)
        previous_col = col
    return (attention_mask, position_ids.to(torch.long))
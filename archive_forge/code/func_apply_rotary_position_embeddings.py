import math
import os
from typing import Optional, Tuple, Union
import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel, SequenceSummary
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
from .configuration_roformer import RoFormerConfig
@staticmethod
def apply_rotary_position_embeddings(sinusoidal_pos, query_layer, key_layer, value_layer=None):
    sin, cos = sinusoidal_pos.chunk(2, dim=-1)
    sin_pos = torch.stack([sin, sin], dim=-1).reshape_as(sinusoidal_pos)
    cos_pos = torch.stack([cos, cos], dim=-1).reshape_as(sinusoidal_pos)
    rotate_half_query_layer = torch.stack([-query_layer[..., 1::2], query_layer[..., ::2]], dim=-1).reshape_as(query_layer)
    query_layer = query_layer * cos_pos + rotate_half_query_layer * sin_pos
    rotate_half_key_layer = torch.stack([-key_layer[..., 1::2], key_layer[..., ::2]], dim=-1).reshape_as(key_layer)
    key_layer = key_layer * cos_pos + rotate_half_key_layer * sin_pos
    if value_layer is not None:
        rotate_half_value_layer = torch.stack([-value_layer[..., 1::2], value_layer[..., ::2]], dim=-1).reshape_as(value_layer)
        value_layer = value_layer * cos_pos + rotate_half_value_layer * sin_pos
        return (query_layer, key_layer, value_layer)
    return (query_layer, key_layer)
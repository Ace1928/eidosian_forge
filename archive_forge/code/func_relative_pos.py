import os
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import numpy as np
import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_funnel import FunnelConfig
def relative_pos(self, pos: torch.Tensor, stride: int, pooled_pos=None, shift: int=1) -> torch.Tensor:
    """
        Build the relative positional vector between `pos` and `pooled_pos`.
        """
    if pooled_pos is None:
        pooled_pos = pos
    ref_point = pooled_pos[0] - pos[0]
    num_remove = shift * len(pooled_pos)
    max_dist = ref_point + num_remove * stride
    min_dist = pooled_pos[0] - pos[-1]
    return torch.arange(max_dist, min_dist - 1, -stride, dtype=torch.long, device=pos.device)
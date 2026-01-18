import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import BackboneOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
from ...utils.backbone_utils import BackboneMixin
from .configuration_dinat import DinatConfig
def maybe_pad(self, hidden_states, height, width):
    window_size = self.window_size
    pad_values = (0, 0, 0, 0, 0, 0)
    if height < window_size or width < window_size:
        pad_l = pad_t = 0
        pad_r = max(0, window_size - width)
        pad_b = max(0, window_size - height)
        pad_values = (0, 0, pad_l, pad_r, pad_t, pad_b)
        hidden_states = nn.functional.pad(hidden_states, pad_values)
    return (hidden_states, pad_values)
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
from torch import nn
from ...modeling_outputs import BaseModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import ModelOutput, add_start_docstrings, logging, replace_return_docstrings
from .configuration_fastspeech2_conformer import (
def shift_relative_position_tensor(self, pos_tensor):
    """
        Args:
            pos_tensor (torch.Tensor of shape (batch_size, head, time1, 2*time1-1)): Input tensor.
        """
    zero_pad = torch.zeros((*pos_tensor.size()[:3], 1), device=pos_tensor.device, dtype=pos_tensor.dtype)
    pos_tensor_padded = torch.cat([zero_pad, pos_tensor], dim=-1)
    pos_tensor_padded = pos_tensor_padded.view(*pos_tensor.size()[:2], pos_tensor.size(3) + 1, pos_tensor.size(2))
    pos_tensor = pos_tensor_padded[:, :, 1:].view_as(pos_tensor)[:, :, :, :pos_tensor.size(-1) // 2 + 1]
    return pos_tensor
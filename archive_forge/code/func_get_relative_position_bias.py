from functools import partial
from typing import Any, Callable, List, Optional, Tuple
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from ...transforms._presets import VideoClassification
from ...utils import _log_api_usage_once
from .._api import register_model, Weights, WeightsEnum
from .._meta import _KINETICS400_CATEGORIES
from .._utils import _ovewrite_named_param, handle_legacy_interface
from ..swin_transformer import PatchMerging, SwinTransformerBlock
def get_relative_position_bias(self, window_size: List[int]) -> torch.Tensor:
    return _get_relative_position_bias(self.relative_position_bias_table, self.relative_position_index, window_size)
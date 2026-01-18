from functools import partial
from typing import Any, Callable, List, Optional, Sequence, Tuple, Type, Union
import torch.nn as nn
from torch import Tensor
from ...transforms._presets import VideoClassification
from ...utils import _log_api_usage_once
from .._api import register_model, Weights, WeightsEnum
from .._meta import _KINETICS400_CATEGORIES
from .._utils import _ovewrite_named_param, handle_legacy_interface
from .._utils import _ModelURLs
@staticmethod
def get_downsample_stride(stride: int) -> Tuple[int, int, int]:
    return (1, stride, stride)
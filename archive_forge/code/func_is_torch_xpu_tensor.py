import pickle
import warnings
from functools import update_wrapper, wraps
from typing import Any, Mapping
import torch
from ..state import PartialState
from .constants import TORCH_DISTRIBUTED_OPERATION_TYPES
from .dataclasses import DistributedType, TensorInformation
from .imports import (
def is_torch_xpu_tensor(tensor):
    return isinstance(tensor, torch.xpu.FloatTensor, torch.xpu.ByteTensor, torch.xpu.IntTensor, torch.xpu.LongTensor, torch.xpu.HalfTensor, torch.xpu.DoubleTensor, torch.xpu.BFloat16Tensor)
import contextlib
import functools
import logging
import os
import warnings
from enum import auto, Enum
from itertools import accumulate, chain
from typing import (
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributed.fsdp._common_utils import (
from torch.distributed.utils import _alloc_storage, _free_storage, _p_assert
from torch.nn.parameter import _ParameterMeta  # type: ignore[attr-defined]
from ._fsdp_extensions import _ext_post_unflatten_transform, _ext_pre_flatten_transform
def flatten_tensors_into_flat_param(self, tensors: List[Tensor], aligned_numel: int, requires_grad: bool) -> FlatParameter:
    flat_param_data = self.flatten_tensors(tensors, aligned_numel)
    return FlatParameter(flat_param_data, requires_grad=requires_grad)
from __future__ import annotations
import operator
import warnings
import weakref
from contextlib import nullcontext
from enum import Enum
from functools import cmp_to_key, reduce
from typing import (
import torch
from torch import sym_float, sym_int, sym_max
def suggest_memory_format(x: TensorLikeType) -> torch.memory_format:
    if x.layout != torch.strided:
        return torch.contiguous_format
    if are_strides_like_channels_last(x.shape, x.stride()):
        return torch.channels_last if x.ndim == 4 else torch.channels_last_3d
    return torch.contiguous_format
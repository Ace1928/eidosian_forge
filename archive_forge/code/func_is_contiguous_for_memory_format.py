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
def is_contiguous_for_memory_format(a: Tensor, *, memory_format: torch.memory_format) -> bool:
    validate_memory_format(memory_format)
    if memory_format == torch.contiguous_format:
        return is_contiguous(a)
    if memory_format == torch.channels_last:
        return is_channels_last_contiguous_2d(a)
    if memory_format == torch.channels_last_3d:
        return is_channels_last_contiguous_3d(a)
    torch._check(False, lambda: f'is_contiguous received unsupported memory format {memory_format}')
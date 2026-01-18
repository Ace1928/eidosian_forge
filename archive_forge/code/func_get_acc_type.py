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
def get_acc_type(dtype: torch.dtype, device: torch.device) -> torch.dtype:
    if device.type == 'cpu':
        return _cpu_acc_type_map.get(dtype, dtype)
    else:
        return get_computation_dtype(dtype)
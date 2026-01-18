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
def are_strides_like_channels_last(shape: Sequence[int], strides: Sequence[int]) -> bool:
    ndim = len(shape)
    if ndim == 4:
        dim_order = [1, 3, 2, 0]
    elif ndim == 5:
        dim_order = [1, 4, 3, 2, 0]
    else:
        return False
    if strides[1] == 0:
        return False
    min = 0
    for d in dim_order:
        if shape[d] == 0:
            return False
        if strides[d] < min:
            return False
        if d == 0 and min == strides[1]:
            return False
        min = strides[d]
        if strides[d] > 1:
            min *= shape[d]
    return True
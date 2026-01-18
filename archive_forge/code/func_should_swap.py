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
def should_swap(idx_a, idx_b):
    for tensor in tensors:
        stride_a = tensor.stride()[idx_a]
        stride_b = tensor.stride()[idx_b]
        if stride_a == 0 or stride_b == 0:
            continue
        if stride_a < stride_b:
            return -1
        if stride_a > stride_b:
            return 1
        if shape[idx_a] > shape[idx_b]:
            return 1
    return 0
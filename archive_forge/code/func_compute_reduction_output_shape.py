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
def compute_reduction_output_shape(shape: ShapeType, dimensions: Sequence) -> Tuple[int, ...]:
    for idx in dimensions:
        validate_idx(len(shape), idx)
    new_shape = []
    for idx in range(len(shape)):
        if idx in dimensions:
            continue
        new_shape.append(shape[idx])
    return tuple(new_shape)
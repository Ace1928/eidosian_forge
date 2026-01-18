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
def extract_shape(*args, allow_cpu_scalar_tensors: bool) -> Optional[ShapeType]:
    shape = None
    scalar_shape = None
    for arg in args:
        if isinstance(arg, Number):
            continue
        elif isinstance(arg, TensorLike):
            if allow_cpu_scalar_tensors and is_cpu_scalar_tensor(arg):
                scalar_shape = arg.shape
                continue
            if shape is None:
                shape = arg.shape
            if not is_same_shape(shape, arg.shape):
                return None
        else:
            return None
    return shape if shape is not None else scalar_shape
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
def check_in_bounds_for_storage(a: torch.TypedStorage, shape: ShapeType, strides: StrideType, storage_offset: int):
    """
    Determines if the given shape, strides, and offset are valid for the given storage.
    """
    required_length = compute_required_storage_length(shape, strides, storage_offset)
    if a.size() < required_length:
        msg = "Can't view a storage of size {} with an offset of {}, shape of {}, and strides of {}, which requires a storage of size {}".format(a.size(), storage_offset, str(shape), str(strides), required_length)
        raise ValueError(msg)
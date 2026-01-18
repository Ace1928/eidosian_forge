from __future__ import annotations
import builtins
import itertools
import operator
from typing import Optional, Sequence
import torch
from . import _dtypes_impl, _util
from ._normalizations import (
from torch import broadcast_shapes
def min_scalar_type(a: ArrayLike, /):
    from ._dtypes import DType
    if a.numel() > 1:
        return DType(a.dtype)
    if a.dtype == torch.bool:
        dtype = torch.bool
    elif a.dtype.is_complex:
        fi = torch.finfo(torch.float32)
        fits_in_single = a.dtype == torch.complex64 or (fi.min <= a.real <= fi.max and fi.min <= a.imag <= fi.max)
        dtype = torch.complex64 if fits_in_single else torch.complex128
    elif a.dtype.is_floating_point:
        for dt in [torch.float16, torch.float32, torch.float64]:
            fi = torch.finfo(dt)
            if fi.min <= a <= fi.max:
                dtype = dt
                break
    else:
        for dt in [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]:
            ii = torch.iinfo(dt)
            if ii.min <= a <= ii.max:
                dtype = dt
                break
    return DType(dtype)
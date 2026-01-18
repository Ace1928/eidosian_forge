from __future__ import annotations  # remove after python 3.11
from functools import wraps
from typing import List, Optional, Sequence, Tuple, TypeVar
from .._C.libtriton.triton import ir
from ..common.build import is_hip
from . import core as tl
def computation_type_impl(a_ty: tl.dtype, b_ty: tl.dtype, div_or_mod: bool) -> tl.dtype:
    if a_ty.is_fp64() or b_ty.is_fp64():
        return tl.float64
    if a_ty.is_fp32() or b_ty.is_fp32():
        return tl.float32
    if a_ty.is_fp16() or b_ty.is_fp16():
        if div_or_mod:
            return tl.float32
        else:
            return tl.float16
    if a_ty.is_bf16() or b_ty.is_bf16():
        if div_or_mod:
            return tl.float32
        if a_ty.is_bf16() and b_ty.is_bf16():
            return tl.bfloat16
        return tl.float32
    if not a_ty.is_int() or not b_ty.is_int():
        assert False
    if div_or_mod and a_ty.int_signedness != b_ty.int_signedness:
        raise ValueError('Cannot use /, #, or % with ' + a_ty.__repr__() + ' and ' + b_ty.__repr__() + ' because they have different signedness;this is unlikely to result in a useful answer. Cast them to the same signedness.')
    return integer_promote_impl(a_ty, b_ty)
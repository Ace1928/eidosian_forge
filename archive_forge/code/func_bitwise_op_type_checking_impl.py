from __future__ import annotations  # remove after python 3.11
from functools import wraps
from typing import List, Optional, Sequence, Tuple, TypeVar
from .._C.libtriton.triton import ir
from ..common.build import is_hip
from . import core as tl
def bitwise_op_type_checking_impl(input: tl.tensor, other: tl.tensor, builder: ir.builder) -> Tuple[tl.tensor, tl.tensor]:
    input, other = binary_op_type_checking_impl(input, other, builder, False, False, False)
    input_sca_ty = input.type.scalar
    other_sca_ty = other.type.scalar
    if not input_sca_ty.is_int() or not other_sca_ty.is_int():
        raise IncompatibleTypeErrorImpl(input_sca_ty, other_sca_ty)
    ret_sca_ty = integer_promote_impl(input_sca_ty, other_sca_ty)
    if ret_sca_ty != input_sca_ty:
        input = cast(input, ret_sca_ty, builder)
    if ret_sca_ty != other_sca_ty:
        other = cast(other, ret_sca_ty, builder)
    return (input, other)
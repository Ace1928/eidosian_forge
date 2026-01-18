from __future__ import annotations  # remove after python 3.11
from functools import wraps
from typing import List, Optional, Sequence, Tuple, TypeVar
from .._C.libtriton.triton import ir
from ..common.build import is_hip
from . import core as tl
def binary_op_type_checking_impl(lhs: tl.tensor, rhs: tl.tensor, builder: ir.builder, allow_lhs_ptr=False, allow_rhs_ptr=False, arithmetic_check=True, div_or_mod=False) -> Tuple[tl.tensor, tl.tensor]:
    lhs, rhs = broadcast_impl_value(lhs, rhs, builder)
    lhs_sca_ty = lhs.type.scalar
    rhs_sca_ty = rhs.type.scalar
    check_ptr_type_impl(lhs_sca_ty, rhs_sca_ty, allow_lhs_ptr)
    check_ptr_type_impl(rhs_sca_ty, lhs_sca_ty, allow_rhs_ptr)
    if arithmetic_check and (not lhs_sca_ty.is_ptr()) and (not rhs_sca_ty.is_ptr()):
        ret_sca_ty = computation_type_impl(lhs_sca_ty, rhs_sca_ty, div_or_mod)
        lhs = cast(lhs, ret_sca_ty, builder)
        rhs = cast(rhs, ret_sca_ty, builder)
    return (lhs, rhs)
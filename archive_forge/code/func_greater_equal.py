from __future__ import annotations  # remove after python 3.11
from functools import wraps
from typing import List, Optional, Sequence, Tuple, TypeVar
from .._C.libtriton.triton import ir
from ..common.build import is_hip
from . import core as tl
def greater_equal(input: tl.tensor, other: tl.tensor, builder: ir.builder) -> tl.tensor:
    input, other = binary_op_type_checking_impl(input, other, builder)
    scalar_ty = input.type.scalar
    if scalar_ty.is_floating():
        return tl.tensor(builder.create_fcmpOGE(input.handle, other.handle), _bool_like(input))
    elif scalar_ty.is_int():
        if scalar_ty.is_int_signed():
            return tl.tensor(builder.create_icmpSGE(input.handle, other.handle), _bool_like(input))
        else:
            return tl.tensor(builder.create_icmpUGE(input.handle, other.handle), _bool_like(input))
    assert False
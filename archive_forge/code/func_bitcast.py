from __future__ import annotations  # remove after python 3.11
from functools import wraps
from typing import List, Optional, Sequence, Tuple, TypeVar
from .._C.libtriton.triton import ir
from ..common.build import is_hip
from . import core as tl
def bitcast(input: tl.tensor, dst_ty: tl.dtype, builder: ir.builder) -> tl.tensor:
    src_ty = input.type
    if src_ty.is_block():
        dst_ty = tl.block_type(dst_ty.scalar, input.type.get_block_shapes())
    if src_ty == dst_ty:
        return input
    src_sca_ty = src_ty.scalar
    dst_sca_ty = dst_ty.scalar
    if src_sca_ty.is_ptr() or dst_sca_ty.is_ptr():
        return cast(input, dst_ty, builder)
    src_bits = src_sca_ty.primitive_bitwidth
    dst_bits = dst_sca_ty.primitive_bitwidth
    if src_bits != dst_bits:
        raise ValueError('Cannot bitcast data-type of size ' + str(src_bits) + ' to data-type of size ' + str(dst_bits))
    return tl.tensor(builder.create_bitcast(input.handle, dst_ty.to_ir(builder)), dst_ty)
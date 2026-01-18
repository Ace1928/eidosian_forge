from __future__ import annotations  # remove after python 3.11
from functools import wraps
from typing import List, Optional, Sequence, Tuple, TypeVar
from .._C.libtriton.triton import ir
from ..common.build import is_hip
from . import core as tl
def broadcast_impl_value(lhs: tl.tensor, rhs: tl.tensor, builder: ir.builder) -> tl.tensor:
    lhs_ty = lhs.type
    rhs_ty = rhs.type
    if lhs_ty.is_block() and (not rhs_ty.is_block()):
        rhs_ty = tl.block_type(rhs_ty.scalar, lhs_ty.shape)
        rhs = tl.tensor(builder.create_splat(rhs.handle, lhs_ty.get_block_shapes()), rhs_ty)
    elif not lhs_ty.is_block() and rhs_ty.is_block():
        lhs_ty = tl.block_type(lhs_ty.scalar, rhs_ty.shape)
        lhs = tl.tensor(builder.create_splat(lhs.handle, rhs_ty.get_block_shapes()), lhs_ty)
    elif lhs_ty.is_block() and rhs_ty.is_block():
        lhs_shape = lhs_ty.get_block_shapes()
        rhs_shape = rhs_ty.get_block_shapes()
        if len(lhs_shape) < len(rhs_shape):
            for dim in range(len(lhs_shape), len(rhs_shape)):
                lhs = tl.tensor(builder.create_expand_dims(lhs.handle, 0), tl.block_type(lhs_ty.scalar, [1] + lhs_shape))
                lhs_ty = lhs.type
                lhs_shape = lhs_ty.get_block_shapes()
        elif len(rhs_shape) < len(lhs_shape):
            for dim in range(len(rhs_shape), len(lhs_shape)):
                rhs = tl.tensor(builder.create_expand_dims(rhs.handle, 0), tl.block_type(rhs_ty.scalar, [1] + rhs_shape))
                rhs_ty = rhs.type
                rhs_shape = rhs_ty.get_block_shapes()
        assert len(rhs_shape) == len(lhs_shape)
        ret_shape = []
        for i, left in enumerate(lhs_shape):
            right = rhs_shape[i]
            if left == 1:
                ret_shape.append(right)
            elif right == 1:
                ret_shape.append(left)
            elif left == right:
                ret_shape.append(left)
            else:
                raise ValueError('Cannot make_shape_compatible: incompatible dimensions at index ' + str(i) + ': ' + str(left) + ' and ' + str(right))
        if lhs_shape != ret_shape:
            ret_ty = tl.block_type(lhs_ty.scalar, ret_shape)
            lhs = tl.tensor(builder.create_broadcast(lhs.handle, ret_shape), ret_ty)
        if rhs_shape != ret_shape:
            ret_ty = tl.block_type(rhs_ty.scalar, ret_shape)
            rhs = tl.tensor(builder.create_broadcast(rhs.handle, ret_shape), ret_ty)
    return (lhs, rhs)
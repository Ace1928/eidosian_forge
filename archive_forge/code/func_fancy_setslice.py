import functools
import math
import operator
from llvmlite import ir
from llvmlite.ir import Constant
import numpy as np
from numba import pndindex, literal_unroll
from numba.core import types, typing, errors, cgutils, extending
from numba.np.numpy_support import (as_dtype, from_dtype, carray, farray,
from numba.np.numpy_support import type_can_asarray, is_nonelike, numpy_version
from numba.core.imputils import (lower_builtin, lower_getattr,
from numba.core.typing import signature
from numba.core.types import StringLiteral
from numba.core.extending import (register_jitable, overload, overload_method,
from numba.misc import quicksort, mergesort
from numba.cpython import slicing
from numba.cpython.unsafe.tuple import tuple_setitem, build_full_slice_tuple
from numba.core.extending import overload_classmethod
from numba.core.typing.npydecl import (parse_dtype as ty_parse_dtype,
def fancy_setslice(context, builder, sig, args, index_types, indices):
    """
    Implement slice assignment for arrays.  This implementation works for
    basic as well as fancy indexing, since there's no functional difference
    between the two for indexed assignment.
    """
    aryty, _, srcty = sig.args
    ary, _, src = args
    ary = make_array(aryty)(context, builder, ary)
    dest_shapes = cgutils.unpack_tuple(builder, ary.shape)
    dest_strides = cgutils.unpack_tuple(builder, ary.strides)
    dest_data = ary.data
    indexer = FancyIndexer(context, builder, aryty, ary, index_types, indices)
    indexer.prepare()
    if isinstance(srcty, types.Buffer):
        src_dtype = srcty.dtype
        index_shape = indexer.get_shape()
        src = make_array(srcty)(context, builder, src)
        srcty, src = _broadcast_to_shape(context, builder, srcty, src, index_shape)
        src_shapes = cgutils.unpack_tuple(builder, src.shape)
        src_strides = cgutils.unpack_tuple(builder, src.strides)
        src_data = src.data
        shape_error = cgutils.false_bit
        assert len(index_shape) == len(src_shapes)
        for u, v in zip(src_shapes, index_shape):
            shape_error = builder.or_(shape_error, builder.icmp_signed('!=', u, v))
        with builder.if_then(shape_error, likely=False):
            msg = 'cannot assign slice from input of different size'
            context.call_conv.return_user_exc(builder, ValueError, (msg,))
        src_start, src_end = get_array_memory_extents(context, builder, srcty, src, src_shapes, src_strides, src_data)
        dest_lower, dest_upper = indexer.get_offset_bounds(dest_strides, ary.itemsize)
        dest_start, dest_end = compute_memory_extents(context, builder, dest_lower, dest_upper, dest_data)
        use_copy = extents_may_overlap(context, builder, src_start, src_end, dest_start, dest_end)
        src_getitem, src_cleanup = maybe_copy_source(context, builder, use_copy, srcty, src, src_shapes, src_strides, src_data)
    elif isinstance(srcty, types.Sequence):
        src_dtype = srcty.dtype
        index_shape = indexer.get_shape()
        assert len(index_shape) == 1
        len_impl = context.get_function(len, signature(types.intp, srcty))
        seq_len = len_impl(builder, (src,))
        shape_error = builder.icmp_signed('!=', index_shape[0], seq_len)
        with builder.if_then(shape_error, likely=False):
            msg = 'cannot assign slice from input of different size'
            context.call_conv.return_user_exc(builder, ValueError, (msg,))

        def src_getitem(source_indices):
            idx, = source_indices
            getitem_impl = context.get_function(operator.getitem, signature(src_dtype, srcty, types.intp))
            return getitem_impl(builder, (src, idx))

        def src_cleanup():
            pass
    else:
        src_dtype = srcty

        def src_getitem(source_indices):
            return src

        def src_cleanup():
            pass
    zero = context.get_constant(types.uintp, 0)
    dest_indices, counts = indexer.begin_loops()
    counts = list(counts)
    for i in indexer.newaxes:
        counts.insert(i, zero)
    source_indices = [c for c in counts if c is not None]
    val = src_getitem(source_indices)
    val = context.cast(builder, val, src_dtype, aryty.dtype)
    dest_ptr = cgutils.get_item_pointer2(context, builder, dest_data, dest_shapes, dest_strides, aryty.layout, dest_indices, wraparound=False, boundscheck=context.enable_boundscheck)
    store_item(context, builder, aryty, val, dest_ptr)
    indexer.end_loops()
    src_cleanup()
    return context.get_dummy_value()
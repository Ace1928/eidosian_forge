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
def fancy_getitem(context, builder, sig, args, aryty, ary, index_types, indices):
    shapes = cgutils.unpack_tuple(builder, ary.shape)
    strides = cgutils.unpack_tuple(builder, ary.strides)
    data = ary.data
    indexer = FancyIndexer(context, builder, aryty, ary, index_types, indices)
    indexer.prepare()
    out_ty = sig.return_type
    out_shapes = indexer.get_shape()
    out = _empty_nd_impl(context, builder, out_ty, out_shapes)
    out_data = out.data
    out_idx = cgutils.alloca_once_value(builder, context.get_constant(types.intp, 0))
    indices, _ = indexer.begin_loops()
    ptr = cgutils.get_item_pointer2(context, builder, data, shapes, strides, aryty.layout, indices, wraparound=False, boundscheck=context.enable_boundscheck)
    val = load_item(context, builder, aryty, ptr)
    cur = builder.load(out_idx)
    ptr = builder.gep(out_data, [cur])
    store_item(context, builder, out_ty, val, ptr)
    next_idx = cgutils.increment_index(builder, cur)
    builder.store(next_idx, out_idx)
    indexer.end_loops()
    return impl_ret_new_ref(context, builder, out_ty, out._getvalue())
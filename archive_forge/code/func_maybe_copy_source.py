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
def maybe_copy_source(context, builder, use_copy, srcty, src, src_shapes, src_strides, src_data):
    ptrty = src_data.type
    copy_layout = 'C'
    copy_data = cgutils.alloca_once_value(builder, src_data)
    copy_shapes = src_shapes
    copy_strides = None
    with builder.if_then(use_copy, likely=False):
        allocsize = builder.mul(src.itemsize, src.nitems)
        data = context.nrt.allocate(builder, allocsize)
        voidptrty = data.type
        data = builder.bitcast(data, ptrty)
        builder.store(data, copy_data)
        intp_t = context.get_value_type(types.intp)
        with cgutils.loop_nest(builder, src_shapes, intp_t) as indices:
            src_ptr = cgutils.get_item_pointer2(context, builder, src_data, src_shapes, src_strides, srcty.layout, indices)
            dest_ptr = cgutils.get_item_pointer2(context, builder, data, copy_shapes, copy_strides, copy_layout, indices)
            builder.store(builder.load(src_ptr), dest_ptr)

    def src_getitem(source_indices):
        src_ptr = cgutils.alloca_once(builder, ptrty)
        with builder.if_else(use_copy, likely=False) as (if_copy, otherwise):
            with if_copy:
                builder.store(cgutils.get_item_pointer2(context, builder, builder.load(copy_data), copy_shapes, copy_strides, copy_layout, source_indices, wraparound=False), src_ptr)
            with otherwise:
                builder.store(cgutils.get_item_pointer2(context, builder, src_data, src_shapes, src_strides, srcty.layout, source_indices, wraparound=False), src_ptr)
        return load_item(context, builder, srcty, builder.load(src_ptr))

    def src_cleanup():
        with builder.if_then(use_copy, likely=False):
            data = builder.load(copy_data)
            data = builder.bitcast(data, voidptrty)
            context.nrt.free(builder, data)
    return (src_getitem, src_cleanup)
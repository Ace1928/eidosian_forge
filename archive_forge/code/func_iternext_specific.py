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
def iternext_specific(self, context, builder, arrty, arr, result):
    ndim = arrty.ndim
    shapes = cgutils.unpack_tuple(builder, arr.shape, ndim)
    strides = cgutils.unpack_tuple(builder, arr.strides, ndim)
    indices = self.indices
    pointers = self.pointers
    zero = context.get_constant(types.intp, 0)
    bbend = builder.append_basic_block('end')
    is_exhausted = cgutils.as_bool_bit(builder, builder.load(self.exhausted))
    with cgutils.if_unlikely(builder, is_exhausted):
        result.set_valid(False)
        builder.branch(bbend)
    result.set_valid(True)
    last_ptr = cgutils.gep_inbounds(builder, pointers, ndim - 1)
    ptr = builder.load(last_ptr)
    value = load_item(context, builder, arrty, ptr)
    if kind == 'flat':
        result.yield_(value)
    else:
        idxvals = [builder.load(cgutils.gep_inbounds(builder, indices, dim)) for dim in range(ndim)]
        idxtuple = cgutils.pack_array(builder, idxvals)
        result.yield_(cgutils.make_anonymous_struct(builder, [idxtuple, value]))
    for dim in reversed(range(ndim)):
        idxptr = cgutils.gep_inbounds(builder, indices, dim)
        idx = cgutils.increment_index(builder, builder.load(idxptr))
        count = shapes[dim]
        stride = strides[dim]
        in_bounds = builder.icmp_signed('<', idx, count)
        with cgutils.if_likely(builder, in_bounds):
            builder.store(idx, idxptr)
            ptrptr = cgutils.gep_inbounds(builder, pointers, dim)
            ptr = builder.load(ptrptr)
            ptr = cgutils.pointer_add(builder, ptr, stride)
            builder.store(ptr, ptrptr)
            for inner_dim in range(dim + 1, ndim):
                ptrptr = cgutils.gep_inbounds(builder, pointers, inner_dim)
                builder.store(ptr, ptrptr)
            builder.branch(bbend)
        builder.store(zero, idxptr)
    builder.store(cgutils.true_byte, self.exhausted)
    builder.branch(bbend)
    builder.position_at_end(bbend)
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
@lower_builtin('array.transpose', types.Array, types.BaseTuple)
def array_transpose_tuple(context, builder, sig, args):
    aryty = sig.args[0]
    ary = make_array(aryty)(context, builder, args[0])
    axisty, axis = (sig.args[1], args[1])
    num_axis, dtype = (axisty.count, axisty.dtype)
    ll_intp = context.get_value_type(types.intp)
    ll_ary_size = ir.ArrayType(ll_intp, num_axis)
    arys = [axis, ary.shape, ary.strides]
    ll_arys = [cgutils.alloca_once(builder, ll_ary_size) for _ in arys]
    for src, dst in zip(arys, ll_arys):
        builder.store(src, dst)
    np_ary_ty = types.Array(dtype=dtype, ndim=1, layout='C')
    np_itemsize = context.get_constant(types.intp, context.get_abi_sizeof(ll_intp))
    np_arys = [make_array(np_ary_ty)(context, builder) for _ in arys]
    for np_ary, ll_ary in zip(np_arys, ll_arys):
        populate_array(np_ary, data=builder.bitcast(ll_ary, ll_intp.as_pointer()), shape=[context.get_constant(types.intp, num_axis)], strides=[np_itemsize], itemsize=np_itemsize, meminfo=None)
    context.compile_internal(builder, permute_arrays, typing.signature(types.void, np_ary_ty, np_ary_ty, np_ary_ty), [a._getvalue() for a in np_arys])
    ret = make_array(sig.return_type)(context, builder)
    populate_array(ret, data=ary.data, shape=builder.load(ll_arys[1]), strides=builder.load(ll_arys[2]), itemsize=ary.itemsize, meminfo=ary.meminfo, parent=ary.parent)
    res = ret._getvalue()
    return impl_ret_borrowed(context, builder, sig.return_type, res)
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
@lower_getattr_generic(types.Array)
def array_record_getattr(context, builder, typ, value, attr):
    """
    Generic getattr() implementation for record arrays: fetch the given
    record member, i.e. a subarray.
    """
    arrayty = make_array(typ)
    array = arrayty(context, builder, value)
    rectype = typ.dtype
    if not isinstance(rectype, types.Record):
        raise NotImplementedError('attribute %r of %s not defined' % (attr, typ))
    dtype = rectype.typeof(attr)
    offset = rectype.offset(attr)
    if isinstance(dtype, types.NestedArray):
        resty = typ.copy(dtype=dtype.dtype, ndim=typ.ndim + dtype.ndim, layout='A')
    else:
        resty = typ.copy(dtype=dtype, layout='A')
    raryty = make_array(resty)
    rary = raryty(context, builder)
    constoffset = context.get_constant(types.intp, offset)
    newdataptr = cgutils.pointer_add(builder, array.data, constoffset, return_type=rary.data.type)
    if isinstance(dtype, types.NestedArray):
        shape = cgutils.unpack_tuple(builder, array.shape, typ.ndim)
        shape += [context.get_constant(types.intp, i) for i in dtype.shape]
        strides = cgutils.unpack_tuple(builder, array.strides, typ.ndim)
        strides += [context.get_constant(types.intp, i) for i in dtype.strides]
        datasize = context.get_abi_sizeof(context.get_data_type(dtype.dtype))
    else:
        shape = array.shape
        strides = array.strides
        datasize = context.get_abi_sizeof(context.get_data_type(dtype))
    populate_array(rary, data=newdataptr, shape=shape, strides=strides, itemsize=context.get_constant(types.intp, datasize), meminfo=array.meminfo, parent=array.parent)
    res = rary._getvalue()
    return impl_ret_borrowed(context, builder, resty, res)
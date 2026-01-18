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
@lower_getattr_generic(types.Record)
def record_getattr(context, builder, typ, value, attr):
    """
    Generic getattr() implementation for records: get the given record member.
    """
    context.sentry_record_alignment(typ, attr)
    offset = typ.offset(attr)
    elemty = typ.typeof(attr)
    if isinstance(elemty, types.NestedArray):
        aryty = make_array(elemty)
        ary = aryty(context, builder)
        dtype = elemty.dtype
        newshape = [context.get_constant(types.intp, s) for s in elemty.shape]
        newstrides = [context.get_constant(types.intp, s) for s in elemty.strides]
        newdata = cgutils.get_record_member(builder, value, offset, context.get_data_type(dtype))
        populate_array(ary, data=newdata, shape=cgutils.pack_array(builder, newshape), strides=cgutils.pack_array(builder, newstrides), itemsize=context.get_constant(types.intp, elemty.size), meminfo=None, parent=None)
        res = ary._getvalue()
        return impl_ret_borrowed(context, builder, typ, res)
    else:
        dptr = cgutils.get_record_member(builder, value, offset, context.get_data_type(elemty))
        align = None if typ.aligned else 1
        res = context.unpack_value(builder, elemty, dptr, align)
        return impl_ret_borrowed(context, builder, typ, res)
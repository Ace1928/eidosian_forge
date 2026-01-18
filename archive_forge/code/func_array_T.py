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
@lower_getattr(types.Array, 'T')
def array_T(context, builder, typ, value):
    if typ.ndim <= 1:
        res = value
    else:
        ary = make_array(typ)(context, builder, value)
        ret = make_array(typ)(context, builder)
        shapes = cgutils.unpack_tuple(builder, ary.shape, typ.ndim)
        strides = cgutils.unpack_tuple(builder, ary.strides, typ.ndim)
        populate_array(ret, data=ary.data, shape=cgutils.pack_array(builder, shapes[::-1]), strides=cgutils.pack_array(builder, strides[::-1]), itemsize=ary.itemsize, meminfo=ary.meminfo, parent=ary.parent)
        res = ret._getvalue()
    return impl_ret_borrowed(context, builder, typ, res)
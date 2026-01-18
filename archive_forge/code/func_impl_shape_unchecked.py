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
@lower_builtin(reshape_unchecked, types.Array, types.BaseTuple, types.BaseTuple)
def impl_shape_unchecked(context, builder, sig, args):
    aryty = sig.args[0]
    retty = sig.return_type
    ary = make_array(aryty)(context, builder, args[0])
    out = make_array(retty)(context, builder)
    shape = cgutils.unpack_tuple(builder, args[1])
    strides = cgutils.unpack_tuple(builder, args[2])
    populate_array(out, data=ary.data, shape=shape, strides=strides, itemsize=ary.itemsize, meminfo=ary.meminfo)
    res = out._getvalue()
    return impl_ret_borrowed(context, builder, retty, res)
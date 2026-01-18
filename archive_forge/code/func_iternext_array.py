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
@lower_builtin('iternext', types.ArrayIterator)
@iternext_impl(RefType.BORROWED)
def iternext_array(context, builder, sig, args, result):
    [iterty] = sig.args
    [iter] = args
    arrayty = iterty.array_type
    iterobj = context.make_helper(builder, iterty, value=iter)
    ary = make_array(arrayty)(context, builder, value=iterobj.array)
    nitems, = cgutils.unpack_tuple(builder, ary.shape, count=1)
    index = builder.load(iterobj.index)
    is_valid = builder.icmp_signed('<', index, nitems)
    result.set_valid(is_valid)
    with builder.if_then(is_valid):
        value = _getitem_array_single_int(context, builder, iterty.yield_type, arrayty, ary, index)
        result.yield_(value)
        nindex = cgutils.increment_index(builder, index)
        builder.store(nindex, iterobj.index)
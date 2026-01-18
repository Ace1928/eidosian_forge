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
@lower_builtin('array.argsort', types.Array, types.StringLiteral)
@lower_builtin(np.argsort, types.Array, types.StringLiteral)
def array_argsort(context, builder, sig, args):
    arytype, kind = sig.args
    sort_func = get_sort_func(kind=kind.literal_value, lt_impl=lt_implementation(arytype.dtype), is_argsort=True)

    def array_argsort_impl(arr):
        return sort_func(arr)
    innersig = sig.replace(args=sig.args[:1])
    innerargs = args[:1]
    return context.compile_internal(builder, array_argsort_impl, innersig, innerargs)
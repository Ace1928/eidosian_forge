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
@overload(np.take)
@overload_method(types.Array, 'take')
def numpy_take(a, indices):
    if isinstance(a, types.Array) and isinstance(indices, types.Integer):

        def take_impl(a, indices):
            if indices > a.size - 1 or indices < -a.size:
                raise IndexError('Index out of bounds')
            return a.ravel()[indices]
        return take_impl
    if all((isinstance(arg, types.Array) for arg in [a, indices])):
        F_order = indices.layout == 'F'

        def take_impl(a, indices):
            ret = np.empty(indices.size, dtype=a.dtype)
            if F_order:
                walker = indices.copy()
            else:
                walker = indices
            it = np.nditer(walker)
            i = 0
            flat = a.ravel()
            for x in it:
                if x > a.size - 1 or x < -a.size:
                    raise IndexError('Index out of bounds')
                ret[i] = flat[x]
                i = i + 1
            return ret.reshape(indices.shape)
        return take_impl
    if isinstance(a, types.Array) and isinstance(indices, (types.List, types.BaseTuple)):

        def take_impl(a, indices):
            convert = np.array(indices)
            ret = np.empty(convert.size, dtype=a.dtype)
            it = np.nditer(convert)
            i = 0
            flat = a.ravel()
            for x in it:
                if x > a.size - 1 or x < -a.size:
                    raise IndexError('Index out of bounds')
                ret[i] = flat[x]
                i = i + 1
            return ret.reshape(convert.shape)
        return take_impl
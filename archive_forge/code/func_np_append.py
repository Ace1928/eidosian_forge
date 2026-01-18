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
@overload(np.append)
def np_append(arr, values, axis=None):
    if not type_can_asarray(arr):
        raise errors.TypingError('The first argument "arr" must be array-like')
    if not type_can_asarray(values):
        raise errors.TypingError('The second argument "values" must be array-like')
    if is_nonelike(axis):

        def impl(arr, values, axis=None):
            arr = np.ravel(np.asarray(arr))
            values = np.ravel(np.asarray(values))
            return np.concatenate((arr, values))
    else:
        if not isinstance(axis, types.Integer):
            raise errors.TypingError('The third argument "axis" must be an integer')

        def impl(arr, values, axis=None):
            return np.concatenate((arr, values), axis=axis)
    return impl
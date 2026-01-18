import math
from collections import namedtuple
import operator
import warnings
import llvmlite.ir
import numpy as np
from numba.core import types, cgutils
from numba.core.extending import overload, overload_method, register_jitable
from numba.np.numpy_support import (as_dtype, type_can_asarray, type_is_scalar,
from numba.core.imputils import (lower_builtin, impl_ret_borrowed,
from numba.np.arrayobj import (make_array, load_item, store_item,
from numba.np.linalg import ensure_blas
from numba.core.extending import intrinsic
from numba.core.errors import (RequireLiteralValue, TypingError,
from numba.cpython.unsafe.tuple import tuple_setitem
@overload(np.average)
def np_average(a, axis=None, weights=None):
    if weights is None or isinstance(weights, types.NoneType):

        def np_average_impl(a, axis=None, weights=None):
            arr = np.asarray(a)
            return np.mean(arr)
    elif axis is None or isinstance(axis, types.NoneType):

        def np_average_impl(a, axis=None, weights=None):
            arr = np.asarray(a)
            weights = np.asarray(weights)
            if arr.shape != weights.shape:
                if axis is None:
                    raise TypeError('Numba does not support average when shapes of a and weights differ.')
                if weights.ndim != 1:
                    raise TypeError('1D weights expected when shapes of a and weights differ.')
            scl = np.sum(weights)
            if scl == 0.0:
                raise ZeroDivisionError("Weights sum to zero, can't be normalized.")
            avg = np.sum(np.multiply(arr, weights)) / scl
            return avg
    else:

        def np_average_impl(a, axis=None, weights=None):
            raise TypeError('Numba does not support average with axis.')
    return np_average_impl
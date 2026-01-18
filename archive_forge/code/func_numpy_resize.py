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
@overload(np.resize)
def numpy_resize(a, new_shape):
    if not type_can_asarray(a):
        msg = 'The argument "a" must be array-like'
        raise errors.TypingError(msg)
    if not (isinstance(new_shape, types.UniTuple) and isinstance(new_shape.dtype, types.Integer) or isinstance(new_shape, types.Integer)):
        msg = 'The argument "new_shape" must be an integer or a tuple of integers'
        raise errors.TypingError(msg)

    def impl(a, new_shape):
        a = np.asarray(a)
        a = np.ravel(a)
        if isinstance(new_shape, tuple):
            new_size = 1
            for dim_length in np.asarray(new_shape):
                new_size *= dim_length
                if dim_length < 0:
                    msg = 'All elements of `new_shape` must be non-negative'
                    raise ValueError(msg)
        else:
            if new_shape < 0:
                msg2 = 'All elements of `new_shape` must be non-negative'
                raise ValueError(msg2)
            new_size = new_shape
        if a.size == 0:
            return np.zeros(new_shape).astype(a.dtype)
        repeats = -(-new_size // a.size)
        res = a
        for i in range(repeats - 1):
            res = np.concatenate((res, a))
        res = res[:new_size]
        return np.reshape(res, new_shape)
    return impl
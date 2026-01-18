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
@overload(np.rot90)
def numpy_rot90(m, k=1):
    if not isinstance(k, (int, types.Integer)):
        raise errors.TypingError('The second argument "k" must be an integer')
    if not isinstance(m, types.Array):
        raise errors.TypingError('The first argument "m" must be an array')
    if m.ndim < 2:
        raise errors.NumbaValueError('Input must be >= 2-d.')

    def impl(m, k=1):
        k = k % 4
        if k == 0:
            return m[:]
        elif k == 1:
            return np.swapaxes(np.fliplr(m), 0, 1)
        elif k == 2:
            return np.flipud(np.fliplr(m))
        elif k == 3:
            return np.fliplr(np.swapaxes(m, 0, 1))
        else:
            raise AssertionError
    return impl
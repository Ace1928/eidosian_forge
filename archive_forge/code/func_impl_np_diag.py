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
@overload(np.diag)
def impl_np_diag(v, k=0):
    if not type_can_asarray(v):
        raise errors.TypingError('The argument "v" must be array-like')
    if isinstance(v, types.Array):
        if v.ndim not in (1, 2):
            raise errors.NumbaTypeError('Input must be 1- or 2-d.')

        def diag_impl(v, k=0):
            if v.ndim == 1:
                s = v.shape
                n = s[0] + abs(k)
                ret = np.zeros((n, n), v.dtype)
                if k >= 0:
                    for i in range(n - k):
                        ret[i, k + i] = v[i]
                else:
                    for i in range(n + k):
                        ret[i - k, i] = v[i]
                return ret
            else:
                rows, cols = v.shape
                if k < 0:
                    rows = rows + k
                if k > 0:
                    cols = cols - k
                n = max(min(rows, cols), 0)
                ret = np.empty(n, v.dtype)
                if k >= 0:
                    for i in range(n):
                        ret[i] = v[i, k + i]
                else:
                    for i in range(n):
                        ret[i] = v[i - k, i]
                return ret
        return diag_impl
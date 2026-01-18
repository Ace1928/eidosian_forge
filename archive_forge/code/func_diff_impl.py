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
def diff_impl(a, n=1):
    if n == 0:
        return a.copy()
    if n < 0:
        raise ValueError('diff(): order must be non-negative')
    size = a.shape[-1]
    out_shape = a.shape[:-1] + (max(size - n, 0),)
    out = np.empty(out_shape, a.dtype)
    if out.size == 0:
        return out
    a2 = a.reshape((-1, size))
    out2 = out.reshape((-1, out.shape[-1]))
    work = np.empty(size, a.dtype)
    for major in range(a2.shape[0]):
        for i in range(size - 1):
            work[i] = a2[major, i + 1] - a2[major, i]
        for niter in range(1, n):
            for i in range(size - niter - 1):
                work[i] = work[i + 1] - work[i]
        out2[major] = work[:size - n]
    return out
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
@register_jitable
def np_interp_impl_inner(x, xp, fp, dtype):
    dz = np.asarray(x, dtype=np.float64)
    dx = np.asarray(xp, dtype=np.float64)
    dy = np.asarray(fp, dtype=np.float64)
    if len(dx) == 0:
        raise ValueError('array of sample points is empty')
    if len(dx) != len(dy):
        raise ValueError('fp and xp are not of the same size.')
    if dx.size == 1:
        return np.full(dz.shape, fill_value=dy[0], dtype=dtype)
    dres = np.empty(dz.shape, dtype=dtype)
    lenx = dz.size
    lenxp = len(dx)
    lval = dy[0]
    rval = dy[lenxp - 1]
    if lenxp == 1:
        xp_val = dx[0]
        fp_val = dy[0]
        for i in range(lenx):
            x_val = dz.flat[i]
            if x_val < xp_val:
                dres.flat[i] = lval
            elif x_val > xp_val:
                dres.flat[i] = rval
            else:
                dres.flat[i] = fp_val
    else:
        j = 0
        if lenxp <= lenx:
            slopes = (dy[1:] - dy[:-1]) / (dx[1:] - dx[:-1])
        else:
            slopes = np.empty(0, dtype=dtype)
        for i in range(lenx):
            x_val = dz.flat[i]
            if np.isnan(x_val):
                dres.flat[i] = x_val
                continue
            j = binary_search_with_guess(x_val, dx, lenxp, j)
            if j == -1:
                dres.flat[i] = lval
            elif j == lenxp:
                dres.flat[i] = rval
            elif j == lenxp - 1:
                dres.flat[i] = dy[j]
            elif dx[j] == x_val:
                dres.flat[i] = dy[j]
            else:
                if slopes.size:
                    slope = slopes[j]
                else:
                    slope = (dy[j + 1] - dy[j]) / (dx[j + 1] - dx[j])
                dres.flat[i] = slope * (x_val - dx[j]) + dy[j]
                if np.isnan(dres.flat[i]):
                    dres.flat[i] = slope * (x_val - dx[j + 1]) + dy[j + 1]
                    if np.isnan(dres.flat[i]) and dy[j] == dy[j + 1]:
                        dres.flat[i] = dy[j]
    return dres
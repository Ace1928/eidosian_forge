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
def np_interp_impl_complex_inner(x, xp, fp, dtype):
    dz = np.asarray(x)
    dx = np.asarray(xp)
    dy = np.asarray(fp)
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
            slopes = np.empty(lenxp - 1, dtype=dtype)
        else:
            slopes = np.empty(0, dtype=dtype)
        if slopes.size:
            for i in range(lenxp - 1):
                inv_dx = 1 / (dx[i + 1] - dx[i])
                real = (dy[i + 1].real - dy[i].real) * inv_dx
                imag = (dy[i + 1].imag - dy[i].imag) * inv_dx
                slopes[i] = real + 1j * imag
        for i in range(lenx):
            x_val = dz.flat[i]
            if np.isnan(x_val):
                real = x_val
                imag = 0.0
                dres.flat[i] = real + 1j * imag
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
                    inv_dx = 1 / (dx[j + 1] - dx[j])
                    real = (dy[j + 1].real - dy[j].real) * inv_dx
                    imag = (dy[j + 1].imag - dy[j].imag) * inv_dx
                    slope = real + 1j * imag
                real = slope.real * (x_val - dx[j]) + dy[j].real
                if np.isnan(real):
                    real = slope.real * (x_val - dx[j + 1]) + dy[j + 1].real
                    if np.isnan(real) and dy[j].real == dy[j + 1].real:
                        real = dy[j].real
                imag = slope.imag * (x_val - dx[j]) + dy[j].imag
                if np.isnan(imag):
                    imag = slope.imag * (x_val - dx[j + 1]) + dy[j + 1].imag
                    if np.isnan(imag) and dy[j].imag == dy[j + 1].imag:
                        imag = dy[j].imag
                dres.flat[i] = real + 1j * imag
    return dres
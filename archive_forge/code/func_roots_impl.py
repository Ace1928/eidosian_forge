import numpy as np
from numpy.polynomial import polynomial as poly
from numpy.polynomial import polyutils as pu
from numba import literal_unroll
from numba.core import types, errors
from numba.core.extending import overload
from numba.np.numpy_support import type_can_asarray, as_dtype, from_dtype
def roots_impl(p):
    if len(p.shape) != 1:
        raise ValueError('Input must be a 1d array.')
    non_zero = np.nonzero(p)[0]
    if len(non_zero) == 0:
        return np.zeros(0, dtype=cast_t)
    tz = len(p) - non_zero[-1] - 1
    p = p[int(non_zero[0]):int(non_zero[-1]) + 1]
    n = len(p)
    if n > 1:
        A = np.diag(np.ones((n - 2,), cast_t), 1).T
        A[0, :] = -p[1:] / p[0]
        roots = np.linalg.eigvals(A)
    else:
        roots = np.zeros(0, dtype=cast_t)
    if tz > 0:
        return np.hstack((roots, np.zeros(tz, dtype=cast_t)))
    else:
        return roots
import numpy as np
from numpy.polynomial import polynomial as poly
from numpy.polynomial import polyutils as pu
from numba import literal_unroll
from numba.core import types, errors
from numba.core.extending import overload
from numba.np.numpy_support import type_can_asarray, as_dtype, from_dtype
@overload(poly.polyint)
def poly_polyint(c, m=1):
    if not type_can_asarray(c):
        msg = 'The argument "c" must be array-like'
        raise errors.TypingError(msg)
    if not isinstance(m, (int, types.Integer)):
        msg = 'The argument "m" must be an integer'
        raise errors.TypingError(msg)
    res_dtype = as_dtype(_poly_result_dtype(c))
    if not np.issubdtype(res_dtype, np.number):
        msg = f'Input dtype must be scalar. Found {res_dtype} instead'
        raise errors.TypingError(msg)
    is1D = np.ndim(c) == 1 or (isinstance(c, (types.List, types.BaseTuple)) and isinstance(c.dtype, types.Number))

    def impl(c, m=1):
        c = np.asarray(c).astype(res_dtype)
        cdt = c.dtype
        for i in range(m):
            n = len(c)
            tmp = np.empty((n + 1,) + c.shape[1:], dtype=cdt)
            tmp[0] = c[0] * 0
            tmp[1] = c[0]
            for j in range(1, n):
                tmp[j + 1] = c[j] / (j + 1)
            c = tmp
        if is1D:
            return pu.trimseq(c)
        else:
            return c
    return impl
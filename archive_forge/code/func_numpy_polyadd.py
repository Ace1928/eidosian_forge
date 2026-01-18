import numpy as np
from numpy.polynomial import polynomial as poly
from numpy.polynomial import polyutils as pu
from numba import literal_unroll
from numba.core import types, errors
from numba.core.extending import overload
from numba.np.numpy_support import type_can_asarray, as_dtype, from_dtype
@overload(poly.polyadd)
def numpy_polyadd(c1, c2):
    if not type_can_asarray(c1):
        msg = 'The argument "c1" must be array-like'
        raise errors.TypingError(msg)
    if not type_can_asarray(c2):
        msg = 'The argument "c2" must be array-like'
        raise errors.TypingError(msg)

    def impl(c1, c2):
        arr1, arr2 = pu.as_series((c1, c2))
        diff = len(arr2) - len(arr1)
        if diff > 0:
            zr = np.zeros(diff)
            arr1 = np.concatenate((arr1, zr))
        if diff < 0:
            zr = np.zeros(-diff)
            arr2 = np.concatenate((arr2, zr))
        val = arr1 + arr2
        return pu.trimseq(val)
    return impl
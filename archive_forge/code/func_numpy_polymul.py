import numpy as np
from numpy.polynomial import polynomial as poly
from numpy.polynomial import polyutils as pu
from numba import literal_unroll
from numba.core import types, errors
from numba.core.extending import overload
from numba.np.numpy_support import type_can_asarray, as_dtype, from_dtype
@overload(poly.polymul)
def numpy_polymul(c1, c2):
    if not type_can_asarray(c1):
        msg = 'The argument "c1" must be array-like'
        raise errors.TypingError(msg)
    if not type_can_asarray(c2):
        msg = 'The argument "c2" must be array-like'
        raise errors.TypingError(msg)

    def impl(c1, c2):
        arr1, arr2 = pu.as_series((c1, c2))
        val = np.convolve(arr1, arr2)
        return pu.trimseq(val)
    return impl
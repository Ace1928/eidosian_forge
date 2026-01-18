from __future__ import annotations
import functools
from typing import (
import numpy as np
from pandas.compat._optional import import_optional_dependency
@functools.cache
def generate_apply_looper(func, nopython=True, nogil=True, parallel=False):
    if TYPE_CHECKING:
        import numba
    else:
        numba = import_optional_dependency('numba')
    nb_compat_func = numba.extending.register_jitable(func)

    @numba.jit(nopython=nopython, nogil=nogil, parallel=parallel)
    def nb_looper(values, axis):
        if axis == 0:
            first_elem = values[:, 0]
            dim0 = values.shape[1]
        else:
            first_elem = values[0]
            dim0 = values.shape[0]
        res0 = nb_compat_func(first_elem)
        buf_shape = (dim0,) + np.atleast_1d(np.asarray(res0)).shape
        if axis == 0:
            buf_shape = buf_shape[::-1]
        buff = np.empty(buf_shape)
        if axis == 1:
            buff[0] = res0
            for i in numba.prange(1, values.shape[0]):
                buff[i] = nb_compat_func(values[i])
        else:
            buff[:, 0] = res0
            for j in numba.prange(1, values.shape[1]):
                buff[:, j] = nb_compat_func(values[:, j])
        return buff
    return nb_looper
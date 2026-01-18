import numba
import numpy as np
@numba.jit(nopython=True)
def numba_take(lut, data):
    return np.take(lut, data)
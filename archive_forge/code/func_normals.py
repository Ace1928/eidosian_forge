import os
import numba as nb
import numpy as np
from cffi import FFI
from numpy.random import PCG64
def normals(n, bit_generator):
    out = np.empty(n)
    for i in range(n):
        out[i] = random_standard_normal(bit_generator)
    return out
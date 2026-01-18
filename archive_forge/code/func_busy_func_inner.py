import signal
import sys
from numba import njit
import numpy as np
@njit(parallel=True)
def busy_func_inner(a, b):
    c = a + b * np.sqrt(a) + np.sqrt(b)
    d = np.sqrt(a + b * np.sqrt(a) + np.sqrt(b))
    return c + d
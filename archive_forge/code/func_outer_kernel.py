from numba import cuda
from numba.cuda.testing import CUDATestCase
import numpy as np
import sys
@cuda.jit(cache=True)
def outer_kernel(r, x, y):
    r[()] = inner(-y[()], x[()])
from numba import cuda
from numba.cuda.testing import CUDATestCase
import numpy as np
import sys
@cuda.jit(cache=True)
def ambiguous_function(r, x):
    r[()] = x[()] + 6
from numba import cuda
import numpy as np
from numba.cuda.testing import skip_on_cudasim, CUDATestCase
import threading
import unittest
@cuda.jit('void(float64[:], float64[:])')
def copy_plus_1(inp, out):
    i = cuda.grid(1)
    if i < out.size:
        out[i] = inp[i] + 1
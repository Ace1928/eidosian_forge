from numba import cuda
from numba.cuda.testing import CUDATestCase
import numpy as np
import sys
@cuda.jit(cache=True)
def cg_usecase_kernel(r, x):
    grid = cuda.cg.this_grid()
    grid.sync()
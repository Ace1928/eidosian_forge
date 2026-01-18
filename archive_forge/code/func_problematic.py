import numpy as np
from numba import cuda
from numba.cuda.testing import unittest, xfail_unless_cudasim, CUDATestCase
from numba.core import config
@cuda.jit(debug=with_debug_mode, opt=with_opt_mode)
def problematic(x, y):
    tid = cuda.threadIdx.x
    ntid = cuda.blockDim.x
    if tid > 12:
        for i in range(ntid):
            y[i] += x[i] // y[i]
    cuda.syncthreads()
    if tid < 17:
        for i in range(ntid):
            x[i] += x[i] // y[i]
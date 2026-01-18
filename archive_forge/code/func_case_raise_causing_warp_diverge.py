import numpy as np
from numba import cuda
from numba.cuda.testing import unittest, xfail_unless_cudasim, CUDATestCase
from numba.core import config
def case_raise_causing_warp_diverge(self, with_debug_mode):
    """Testing issue #2655.

        Exception raising code can cause the compiler to miss location
        of unifying branch target and resulting in unexpected warp
        divergence.
        """
    with_opt_mode = not with_debug_mode

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

    @cuda.jit
    def oracle(x, y):
        tid = cuda.threadIdx.x
        ntid = cuda.blockDim.x
        if tid > 12:
            for i in range(ntid):
                if y[i] != 0:
                    y[i] += x[i] // y[i]
        cuda.syncthreads()
        if tid < 17:
            for i in range(ntid):
                if y[i] != 0:
                    x[i] += x[i] // y[i]
    n = 32
    got_x = 1.0 / (np.arange(n) + 0.01)
    got_y = 1.0 / (np.arange(n) + 0.01)
    problematic[1, n](got_x, got_y)
    expect_x = 1.0 / (np.arange(n) + 0.01)
    expect_y = 1.0 / (np.arange(n) + 0.01)
    oracle[1, n](expect_x, expect_y)
    np.testing.assert_almost_equal(expect_x, got_x)
    np.testing.assert_almost_equal(expect_y, got_y)
import numpy as np
from numba import cuda, int32, int64, float32, float64
from numba.cuda.testing import unittest, CUDATestCase, skip_on_cudasim
from numba.core import config
def test_lanemask_lt(self):

    @cuda.jit
    def use_lanemask_lt(x):
        i = cuda.grid(1)
        x[i] = cuda.lanemask_lt()
    out = np.zeros(32, dtype=np.uint32)
    use_lanemask_lt[1, 32](out)
    expected = np.asarray([2 ** i - 1 for i in range(32)], dtype=np.uint32)
    np.testing.assert_equal(expected, out)
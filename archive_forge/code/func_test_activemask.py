import numpy as np
from numba import cuda, int32, int64, float32, float64
from numba.cuda.testing import unittest, CUDATestCase, skip_on_cudasim
from numba.core import config
def test_activemask(self):

    @cuda.jit
    def use_activemask(x):
        i = cuda.grid(1)
        if i % 2 == 0:
            x[i] = cuda.activemask()
        else:
            x[i] = cuda.activemask()
    out = np.zeros(32, dtype=np.uint32)
    use_activemask[1, 32](out)
    expected = np.tile((1431655765, 2863311530), 16)
    np.testing.assert_equal(expected, out)
import numpy as np
from numba.cuda.testing import (skip_unless_cc_53,
from numba.np import numpy_support
from numba import cuda, float32, float64, int32, vectorize, void, int64
import math
@skip_on_cudasim('math.remainder(0, 0) raises a ValueError on CUDASim')
def test_math_remainder_0_0(self):

    @cuda.jit(void(float64[::1], int64, int64))
    def test_0_0(r, x, y):
        r[0] = math.remainder(x, y)
    r = np.zeros(1, np.float64)
    test_0_0[1, 1](r, 0, 0)
    self.assertTrue(np.isnan(r[0]))
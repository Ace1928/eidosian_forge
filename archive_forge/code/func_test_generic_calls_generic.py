from numba import cuda, njit
from numba.core.extending import overload
from numba.cuda.testing import CUDATestCase, skip_on_cudasim, unittest
import numpy as np
def test_generic_calls_generic(self):

    def kernel(x):
        generic_calls_generic(x)
    expected = GENERIC_CALLS_GENERIC * GENERIC_FUNCTION_1
    self.check_overload(kernel, expected)
from numba import cuda, njit
from numba.core.extending import overload
from numba.cuda.testing import CUDATestCase, skip_on_cudasim, unittest
import numpy as np
def test_call_two_generic_calls(self):

    def kernel(x):
        generic_func_1(x)
        generic_func_2(x)
    expected = GENERIC_FUNCTION_1 * GENERIC_FUNCTION_2
    self.check_overload(kernel, expected)
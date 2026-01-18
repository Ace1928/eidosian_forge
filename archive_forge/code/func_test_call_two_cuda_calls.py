from numba import cuda, njit
from numba.core.extending import overload
from numba.cuda.testing import CUDATestCase, skip_on_cudasim, unittest
import numpy as np
def test_call_two_cuda_calls(self):

    def kernel(x):
        cuda_func_1(x)
        cuda_func_2(x)
    expected = CUDA_FUNCTION_1 * CUDA_FUNCTION_2
    self.check_overload(kernel, expected)
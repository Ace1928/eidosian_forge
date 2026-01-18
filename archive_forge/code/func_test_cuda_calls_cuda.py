from numba import cuda, njit
from numba.core.extending import overload
from numba.cuda.testing import CUDATestCase, skip_on_cudasim, unittest
import numpy as np
def test_cuda_calls_cuda(self):

    def kernel(x):
        cuda_calls_cuda(x)
    expected = CUDA_CALLS_CUDA * CUDA_FUNCTION_1
    self.check_overload(kernel, expected)
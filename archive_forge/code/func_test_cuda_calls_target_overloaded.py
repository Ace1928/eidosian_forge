from numba import cuda, njit
from numba.core.extending import overload
from numba.cuda.testing import CUDATestCase, skip_on_cudasim, unittest
import numpy as np
def test_cuda_calls_target_overloaded(self):

    def kernel(x):
        cuda_calls_target_overloaded(x)
    expected = CUDA_CALLS_TARGET_OL * CUDA_TARGET_OL
    self.check_overload(kernel, expected)
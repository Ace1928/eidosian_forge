from numba.tests.support import override_config
from numba.cuda.testing import skip_on_cudasim
from numba import cuda
from numba.core import types
from numba.cuda.testing import CUDATestCase
import itertools
import re
import unittest
def test_debug_function_calls_device_function(self):

    @cuda.jit(device=True, debug=True, opt=0)
    def threadid():
        return cuda.blockDim.x * cuda.blockIdx.x + cuda.threadIdx.x

    @cuda.jit((types.int32[:],), debug=True, opt=0)
    def kernel(arr):
        i = cuda.grid(1)
        if i < len(arr):
            arr[i] = threadid()
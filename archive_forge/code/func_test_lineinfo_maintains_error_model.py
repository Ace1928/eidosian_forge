from numba import cuda, float32, int32
from numba.core.errors import NumbaInvalidConfigWarning
from numba.cuda.testing import CUDATestCase, skip_on_cudasim
from numba.tests.support import ignore_internal_warnings
import re
import unittest
import warnings
def test_lineinfo_maintains_error_model(self):
    sig = (float32[::1], float32[::1])

    @cuda.jit(sig, lineinfo=True)
    def divide_kernel(x, y):
        x[0] /= y[0]
    llvm = divide_kernel.inspect_llvm(sig)
    self.assertNotIn('ret i32 1', llvm)
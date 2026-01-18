from numba import cuda, float32, int32
from numba.core.errors import NumbaInvalidConfigWarning
from numba.cuda.testing import CUDATestCase, skip_on_cudasim
from numba.tests.support import ignore_internal_warnings
import re
import unittest
import warnings
def test_lineinfo_in_asm(self):

    @cuda.jit(lineinfo=True)
    def foo(x):
        x[0] = 1
    self._check(foo, sig=(int32[:],), expect=True)
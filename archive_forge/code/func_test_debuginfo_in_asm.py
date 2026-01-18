from numba.tests.support import override_config
from numba.cuda.testing import skip_on_cudasim
from numba import cuda
from numba.core import types
from numba.cuda.testing import CUDATestCase
import itertools
import re
import unittest
def test_debuginfo_in_asm(self):

    @cuda.jit(debug=True, opt=False)
    def foo(x):
        x[0] = 1
    self._check(foo, sig=(types.int32[:],), expect=True)
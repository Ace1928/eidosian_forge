from numba.tests.support import override_config
from numba.cuda.testing import skip_on_cudasim
from numba import cuda
from numba.core import types
from numba.cuda.testing import CUDATestCase
import itertools
import re
import unittest
def test_debug_function_calls_internal_impl(self):

    @cuda.jit((types.int32[:], types.int32[:]), debug=True, opt=False)
    def f(inp, outp):
        outp[0] = 1 if inp[0] in (2, 3) else 3
from numba.tests.support import override_config
from numba.cuda.testing import skip_on_cudasim
from numba import cuda
from numba.core import types
from numba.cuda.testing import CUDATestCase
import itertools
import re
import unittest
def test_environment_override(self):
    with override_config('CUDA_DEBUGINFO_DEFAULT', 1):

        @cuda.jit(opt=False)
        def foo(x):
            x[0] = 1
        self._check(foo, sig=(types.int32[:],), expect=True)

        @cuda.jit(debug=False)
        def bar(x):
            x[0] = 1
        self._check(bar, sig=(types.int32[:],), expect=False)
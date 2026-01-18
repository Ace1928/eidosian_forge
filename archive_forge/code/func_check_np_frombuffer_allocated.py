from itertools import product, cycle
import gc
import sys
import unittest
import warnings
import numpy as np
from numba import jit, njit, typeof
from numba.core import types
from numba.core.errors import TypingError, NumbaValueError
from numba.np.numpy_support import as_dtype, numpy_version
from numba.tests.support import TestCase, MemoryLeakMixin, needs_blas
def check_np_frombuffer_allocated(self, pyfunc):
    cfunc = njit(pyfunc)

    def check(shape):
        expected = pyfunc(shape)
        got = cfunc(shape)
        self.assertPreciseEqual(got, expected)
    check((16,))
    check((4, 4))
    check((1, 0, 1))
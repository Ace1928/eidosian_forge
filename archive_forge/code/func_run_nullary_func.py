import unittest
import numpy as np
from numba import jit, njit
from numba.core import types, errors
from numba.tests.support import TestCase, MemoryLeakMixin
from numba.np import numpy_support
def run_nullary_func(self, pyfunc, flags):
    cfunc = jit((), **flags)(pyfunc)
    expected = pyfunc()
    self.assertPreciseEqual(cfunc(), expected)
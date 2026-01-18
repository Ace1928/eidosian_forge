import unittest
import numpy as np
from numba import jit, njit
from numba.core import types, errors
from numba.tests.support import TestCase, MemoryLeakMixin
from numba.np import numpy_support
def test_float_tuple_iter(self, flags=force_pyobj_flags):
    self.run_nullary_func(float_tuple_iter_usecase, flags)
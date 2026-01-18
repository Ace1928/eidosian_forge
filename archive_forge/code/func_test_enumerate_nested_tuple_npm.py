import unittest
import numpy as np
from numba import jit, njit
from numba.core import types, errors
from numba.tests.support import TestCase, MemoryLeakMixin
from numba.np import numpy_support
def test_enumerate_nested_tuple_npm(self):
    self.test_enumerate_nested_tuple(flags=no_pyobj_flags)
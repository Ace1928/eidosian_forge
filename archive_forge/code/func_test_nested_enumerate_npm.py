import unittest
import numpy as np
from numba import jit, njit
from numba.core import types, errors
from numba.tests.support import TestCase, MemoryLeakMixin
from numba.np import numpy_support
def test_nested_enumerate_npm(self):
    self.test_nested_enumerate(flags=no_pyobj_flags)
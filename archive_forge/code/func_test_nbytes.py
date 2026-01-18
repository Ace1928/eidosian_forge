import array
import numpy as np
from numba import jit
from numba.tests.support import TestCase, compile_function, MemoryLeakMixin
import unittest
def test_nbytes(self):
    for arr in self._arrays():
        m = memoryview(arr)
        self.assertPreciseEqual(nbytes_usecase(m), arr.size * arr.itemsize)
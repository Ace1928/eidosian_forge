import array
import numpy as np
from numba import jit
from numba.tests.support import TestCase, compile_function, MemoryLeakMixin
import unittest
def test_contiguous(self):
    m = memoryview(bytearray(b'xyz'))
    self.assertIs(contiguous_usecase(m), True)
    self.assertIs(c_contiguous_usecase(m), True)
    self.assertIs(f_contiguous_usecase(m), True)
    for arr in self._arrays():
        m = memoryview(arr)
        self.assertIs(contiguous_usecase(m), arr.flags.f_contiguous or arr.flags.c_contiguous)
        self.assertIs(c_contiguous_usecase(m), arr.flags.c_contiguous)
        self.assertIs(f_contiguous_usecase(m), arr.flags.f_contiguous)
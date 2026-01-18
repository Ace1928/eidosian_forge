import gc
import itertools
import numpy as np
import unittest
from numba import jit, njit
from numba.core import types
from numba.tests.support import TestCase
from numba.np import numpy_support
def test_negative_to_unsigned(self):

    def f(x):
        return x
    with self.assertRaises(OverflowError):
        jit('uintp(uintp)', nopython=True)(f)(-5)
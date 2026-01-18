import math
import warnings
from numba import jit
from numba.core.errors import TypingError, NumbaWarning
from numba.tests.support import TestCase
import unittest
def test_mutual_1(self):
    from numba.tests.recursion_usecases import outer_fac
    expect = math.factorial(10)
    self.assertPreciseEqual(outer_fac(10), expect)
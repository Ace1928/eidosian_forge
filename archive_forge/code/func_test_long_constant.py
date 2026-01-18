import numpy as np
import unittest
from numba import jit
from numba.core import utils
from numba.tests.support import TestCase
def test_long_constant(self):
    pyfunc = long_constant
    cfunc = jit((), forceobj=True)(pyfunc)
    self.assertPreciseEqual(pyfunc(12), cfunc(12))
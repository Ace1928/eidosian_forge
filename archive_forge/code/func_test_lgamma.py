import itertools
import math
import sys
import unittest
import warnings
import numpy as np
from numba import njit, types
from numba.tests.support import TestCase
from numba.np import numpy_support
def test_lgamma(self):
    pyfunc = lgamma
    x_values = [1.0, -0.9, -0.1, 0.1, 200.0, 10000000000.0, 1e+30, float('inf')]
    x_types = [types.float32, types.float64] * (len(x_values) // 2)
    self.run_unary(pyfunc, x_types, x_values, prec='double')
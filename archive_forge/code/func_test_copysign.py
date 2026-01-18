import itertools
import math
import sys
import unittest
import warnings
import numpy as np
from numba import njit, types
from numba.tests.support import TestCase
from numba.np import numpy_support
def test_copysign(self):
    pyfunc = copysign
    value_types = [types.float32, types.float64]
    values = [-2, -1, -0.0, 0.0, 1, 2, float('-inf'), float('inf'), float('nan')]
    x_types, x_values, y_values = list(zip(*itertools.product(value_types, values, values)))
    self.run_binary(pyfunc, x_types, x_values, y_values)
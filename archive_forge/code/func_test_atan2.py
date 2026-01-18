import itertools
import math
import sys
import unittest
import warnings
import numpy as np
from numba import njit, types
from numba.tests.support import TestCase
from numba.np import numpy_support
def test_atan2(self):
    pyfunc = atan2
    x_types = [types.int16, types.int32, types.int64, types.uint16, types.uint32, types.uint64, types.float32, types.float64]
    x_values = [-2, -1, -2, 2, 1, 2, 0.1, 0.2]
    y_values = [x * 2 for x in x_values]
    self.run_binary(pyfunc, x_types, x_values, y_values)
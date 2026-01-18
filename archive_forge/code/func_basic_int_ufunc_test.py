import functools
import numpy as np
import unittest
from numba import config, cuda, types
from numba.tests.support import TestCase
from numba.tests.test_ufuncs import BasicUFuncTest
def basic_int_ufunc_test(self, name=None):
    skip_inputs = [types.float32, types.float64, types.Array(types.float32, 1, 'C'), types.Array(types.float32, 2, 'C'), types.Array(types.float64, 1, 'C'), types.Array(types.float64, 2, 'C'), types.Array(types.float64, 3, 'C'), types.Array(types.float64, 2, 'F'), types.complex64, types.complex128, types.Array(types.complex64, 1, 'C'), types.Array(types.complex64, 2, 'C'), types.Array(types.complex128, 1, 'C'), types.Array(types.complex128, 2, 'C')]
    self.basic_ufunc_test(name, skip_inputs=skip_inputs)
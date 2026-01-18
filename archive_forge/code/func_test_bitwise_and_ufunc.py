import functools
import numpy as np
import unittest
from numba import config, cuda, types
from numba.tests.support import TestCase
from numba.tests.test_ufuncs import BasicUFuncTest
def test_bitwise_and_ufunc(self):
    self.basic_int_ufunc_test(np.bitwise_and)
import functools
import numpy as np
import unittest
from numba import config, cuda, types
from numba.tests.support import TestCase
from numba.tests.test_ufuncs import BasicUFuncTest
def test_arctan2_ufunc(self):
    self.basic_ufunc_test(np.arctan2, kinds='f')
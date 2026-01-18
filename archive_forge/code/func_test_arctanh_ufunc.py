import functools
import numpy as np
import unittest
from numba import config, cuda, types
from numba.tests.support import TestCase
from numba.tests.test_ufuncs import BasicUFuncTest
def test_arctanh_ufunc(self):
    to_skip = [types.Array(types.uint32, 1, 'C'), types.uint32, types.Array(types.int32, 1, 'C'), types.int32, types.Array(types.uint64, 1, 'C'), types.uint64, types.Array(types.int64, 1, 'C'), types.int64]
    self.basic_ufunc_test(np.arctanh, skip_inputs=to_skip, kinds='cf')
import unittest
from functools import reduce
import numpy as np
from numba import njit, typeof, prange, pndindex
import numba.parfors.parfor
from numba.core import (
from numba.core.registry import cpu_target
from numba.tests.support import (TestCase, is_parfors_unsupported)
def test_dtype_conversion(self):

    def test_impl(a):
        b = np.ones(20, dtype=a.dtype)
        return b
    arr = np.arange(10)
    args = (arr,)
    argtypes = [typeof(x) for x in args]
    pre_pass = self.run_parfor_pre_pass(test_impl, argtypes)
    self.assertEqual(pre_pass.stats['replaced_func'], 0)
    self.assertEqual(pre_pass.stats['replaced_dtype'], 1)
    self.run_parallel(test_impl, *args)
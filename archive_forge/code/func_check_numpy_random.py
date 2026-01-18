import unittest
from functools import reduce
import numpy as np
from numba import njit, typeof, prange, pndindex
import numba.parfors.parfor
from numba.core import (
from numba.core.registry import cpu_target
from numba.tests.support import (TestCase, is_parfors_unsupported)
def check_numpy_random(self, fn):

    def test_impl():
        n = 10
        a = fn(n)
        return a
    sub_pass = self.run_parfor_sub_pass(test_impl, ())
    self.assertEqual(len(sub_pass.rewritten), 1)
    [record] = sub_pass.rewritten
    self.assertEqual(record['reason'], 'numpy_allocator')
    self.check_records(sub_pass.rewritten)
    self.run_parallel_check_output_array(test_impl)
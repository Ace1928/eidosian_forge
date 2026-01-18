import unittest
from functools import reduce
import numpy as np
from numba import njit, typeof, prange, pndindex
import numba.parfors.parfor
from numba.core import (
from numba.core.registry import cpu_target
from numba.tests.support import (TestCase, is_parfors_unsupported)
def run_parallel_check_output_array(self, func, *args, **kwargs):
    if is_parfors_unsupported:
        return
    expect, got = self._run_parallel(func, *args, **kwargs)
    self.assertIsInstance(expect, np.ndarray)
    self.assertIsInstance(got, np.ndarray)
    self.assertEqual(expect.shape, got.shape)
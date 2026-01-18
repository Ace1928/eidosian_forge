import math
import numpy as np
from numba.tests.support import captured_stdout, override_config
from numba import njit, vectorize, guvectorize
import unittest
def test_jit_subset_behaviour(self):

    def foo(x, y):
        return x - y + y
    fastfoo = njit(fastmath={'reassoc', 'nsz'})(foo)
    slowfoo = njit(fastmath={'reassoc'})(foo)
    self.assertEqual(fastfoo(0.5, np.inf), 0.5)
    self.assertTrue(np.isnan(slowfoo(0.5, np.inf)))
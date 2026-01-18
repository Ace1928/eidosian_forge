import contextlib
import sys
import numpy as np
from numba import vectorize, guvectorize
from numba.tests.support import (TestCase, CheckWarningsMixin,
import unittest
def check_ufunc_raise(self, **vectorize_args):
    f = vectorize(['float64(float64)'], **vectorize_args)(sqrt)
    arr = np.array([1, 4, -2, 9, -1, 16], dtype=np.float64)
    out = np.zeros_like(arr)
    with self.assertRaises(ValueError) as cm:
        f(arr, out)
    self.assertIn('Value must be positive', str(cm.exception))
    self.assertEqual(list(out), [1, 2, 0, 3, 0, 4])
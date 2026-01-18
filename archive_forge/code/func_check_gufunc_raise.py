import contextlib
import sys
import numpy as np
from numba import vectorize, guvectorize
from numba.tests.support import (TestCase, CheckWarningsMixin,
import unittest
def check_gufunc_raise(self, **vectorize_args):
    f = guvectorize(['int32[:], int32[:], int32[:]'], '(n),()->(n)', **vectorize_args)(gufunc_foo)
    arr = np.array([1, 2, -3, 4], dtype=np.int32)
    out = np.zeros_like(arr)
    with self.assertRaises(ValueError) as cm:
        f(arr, 2, out)
    self.assertEqual(list(out), [2, 4, 0, 0])
import contextlib
import sys
import numpy as np
from numba import vectorize, guvectorize
from numba.tests.support import (TestCase, CheckWarningsMixin,
import unittest
def check_divmod_float(self, pyfunc, values, messages):
    """
        Test 1 // 0 and 0 // 0.
        """
    f = vectorize(nopython=True)(pyfunc)
    a = np.array([5.0, 6.0, 0.0, 9.0])
    b = np.array([1.0, 0.0, 0.0, 4.0])
    expected = np.array(values)
    with self.check_warnings(messages):
        res = f(a, b)
        self.assertPreciseEqual(res, expected)
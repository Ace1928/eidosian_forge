import sys
import unittest
from numba.tests.support import captured_stdout
from numba.core.config import IS_WIN32
def test_guvectorize_scalar_return(self):
    with captured_stdout():
        from numba import guvectorize, int64
        import numpy as np

        @guvectorize([(int64[:], int64, int64[:])], '(n),()->()')
        def g(x, y, res):
            acc = 0
            for i in range(x.shape[0]):
                acc += x[i] + y
            res[0] = acc
        a = np.arange(5)
        result = g(a, 2)
        self.assertIsInstance(result, np.integer)
        self.assertEqual(result, 20)
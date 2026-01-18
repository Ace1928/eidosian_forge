from itertools import product, combinations_with_replacement
import numpy as np
from numba import jit, njit, typeof
from numba.np.numpy_support import numpy_version
from numba.tests.support import TestCase, MemoryLeakMixin, tag
import unittest
def check_quantile_exceptions(self, pyfunc):
    cfunc = jit(nopython=True)(pyfunc)

    def check_err(a, q):
        with self.assertRaises(ValueError) as raises:
            cfunc(a, q)
        self.assertEqual('Quantiles must be in the range [0, 1]', str(raises.exception))
    self.disable_leak_check()
    a = np.arange(5)
    check_err(a, -0.5)
    check_err(a, (0.1, 0.1, 1.05))
    check_err(a, (0.1, 0.1, np.nan))
    with self.assertTypingError() as e:
        a = np.arange(5) * 1j
        q = 0.1
        cfunc(a, q)
    self.assertIn('Not supported for complex dtype', str(e.exception))
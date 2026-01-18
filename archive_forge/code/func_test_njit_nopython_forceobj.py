import warnings
import numba
from numba import jit, njit
from numba.tests.support import TestCase, always_test
import unittest
def test_njit_nopython_forceobj(self):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always', RuntimeWarning)
        njit(forceobj=True)
    self.assertEqual(len(w), 1)
    self.assertIn('forceobj is set for njit and is ignored', str(w[0].message))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always', RuntimeWarning)
        njit(nopython=True)
    self.assertEqual(len(w), 1)
    self.assertIn('nopython is set for njit and is ignored', str(w[0].message))

    def py_func(x):
        return x
    jit_func = njit(nopython=True)(py_func)
    jit_func(1)
    self.assertEqual(len(jit_func.nopython_signatures), 1)
    jit_func = njit(forceobj=True)(py_func)
    jit_func(1)
    self.assertEqual(len(jit_func.nopython_signatures), 1)
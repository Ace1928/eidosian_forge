import numpy as np
from numba.np.ufunc.ufuncbuilder import GUFuncBuilder
from numba import vectorize, guvectorize
from numba.np.ufunc import PyUFunc_One
from numba.np.ufunc.dufunc import DUFunc as UFuncBuilder
from numba.tests.support import tag, TestCase
from numba.core import config
import unittest
def test_nested_call_explicit(self):
    """
        Check nested call to an explicitly-typed ufunc.
        """
    from numba.tests.npyufunc.ufuncbuilding_usecases import outer_explicit
    builder = UFuncBuilder(outer_explicit, targetoptions={'nopython': True})
    builder.add('(int64, int64)')
    ufunc = builder.build_ufunc()
    self.assertEqual(ufunc(-1, 3), 2)
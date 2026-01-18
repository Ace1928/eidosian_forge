import numpy as np
from numba.np.ufunc.ufuncbuilder import GUFuncBuilder
from numba import vectorize, guvectorize
from numba.np.ufunc import PyUFunc_One
from numba.np.ufunc.dufunc import DUFunc as UFuncBuilder
from numba.tests.support import tag, TestCase
from numba.core import config
import unittest
def test_basic_gufunc(self):
    from numba.tests.npyufunc.ufuncbuilding_usecases import guadd
    gufb = GUFuncBuilder(guadd, '(x, y),(x, y)->(x, y)')
    cres = gufb.add('void(int32[:,:], int32[:,:], int32[:,:])')
    self.assertFalse(cres.objectmode)
    ufunc = gufb.build_ufunc()
    a = np.arange(10, dtype='int32').reshape(2, 5)
    b = ufunc(a, a)
    self.assertPreciseEqual(a + a, b)
    self.assertEqual(b.dtype, np.dtype('int32'))
    self.assertEqual(ufunc.__name__, 'guadd')
    self.assertIn('A generalized addition', ufunc.__doc__)
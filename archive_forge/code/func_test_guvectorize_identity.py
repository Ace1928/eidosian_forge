import numpy as np
from numba.np.ufunc.ufuncbuilder import GUFuncBuilder
from numba import vectorize, guvectorize
from numba.np.ufunc import PyUFunc_One
from numba.np.ufunc.dufunc import DUFunc as UFuncBuilder
from numba.tests.support import tag, TestCase
from numba.core import config
import unittest
def test_guvectorize_identity(self):
    from numba.tests.npyufunc.ufuncbuilding_usecases import add, guadd
    args = (['(int32[:,:], int32[:,:], int32[:,:])'], '(x,y),(x,y)->(x,y)')
    for identity in self._supported_identities:
        ufunc = guvectorize(*args, identity=identity)(guadd)
        expected = None if identity == 'reorderable' else identity
        self.assertEqual(ufunc.identity, expected)
    ufunc = guvectorize(*args)(guadd)
    self.assertIs(ufunc.identity, None)
    with self.assertRaises(ValueError):
        guvectorize(*args, identity='none')(add)
    with self.assertRaises(ValueError):
        guvectorize(*args, identity=2)(add)
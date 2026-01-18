import numpy as np
from numba.np.ufunc.ufuncbuilder import GUFuncBuilder
from numba import vectorize, guvectorize
from numba.np.ufunc import PyUFunc_One
from numba.np.ufunc.dufunc import DUFunc as UFuncBuilder
from numba.tests.support import tag, TestCase
from numba.core import config
import unittest
def test_guvectorize_invalid_layout(self):
    from numba.tests.npyufunc.ufuncbuilding_usecases import guadd
    sigs = ['(int32[:,:], int32[:,:], int32[:,:])']
    with self.assertRaises(ValueError) as raises:
        guvectorize(sigs, ')-:')(guadd)
    self.assertIn('bad token in signature', str(raises.exception))
    with self.assertRaises(NameError) as raises:
        guvectorize(sigs, '(x,y),(x,y)->(x,z,v)')(guadd)
    self.assertEqual(str(raises.exception), 'undefined output symbols: v,z')
    with self.assertRaises(ValueError) as raises:
        guvectorize(sigs, '(x,y),(x,y),(x,y)->')(guadd)
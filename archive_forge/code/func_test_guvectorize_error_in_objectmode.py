import numpy as np
from numba.np.ufunc.ufuncbuilder import GUFuncBuilder
from numba import vectorize, guvectorize
from numba.np.ufunc import PyUFunc_One
from numba.np.ufunc.dufunc import DUFunc as UFuncBuilder
from numba.tests.support import tag, TestCase
from numba.core import config
import unittest
def test_guvectorize_error_in_objectmode(self):
    from numba.tests.npyufunc.ufuncbuilding_usecases import guerror, MyException
    ufunc = guvectorize(['(int32[:,:], int32[:,:], int32[:,:])'], '(x,y),(x,y)->(x,y)', forceobj=True)(guerror)
    a = np.arange(10, dtype='int32').reshape(2, 5)
    with self.assertRaises(MyException):
        ufunc(a, a)
import numpy as np
from numba import jit, njit, errors
from numba.extending import register_jitable
from numba.tests import usecases
import unittest
def check_two_global_rec_arrs(self, **jitargs):
    ctestfunc = jit(**jitargs)(global_two_rec_arrs)
    arr1 = np.zeros(rec_X.shape, dtype=np.int32)
    arr2 = np.zeros(rec_X.shape, dtype=np.float32)
    arr3 = np.zeros(rec_Y.shape, dtype=np.int16)
    arr4 = np.zeros(rec_Y.shape, dtype=np.float64)
    ctestfunc(arr1, arr2, arr3, arr4)
    np.testing.assert_equal(arr1, rec_X.a)
    np.testing.assert_equal(arr2, rec_X.b)
    np.testing.assert_equal(arr3, rec_Y.c)
    np.testing.assert_equal(arr4, rec_Y.d)
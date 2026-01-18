import sys
from itertools import product
import numpy as np
import unittest
from numba.core import types
from numba.core.errors import NumbaNotImplementedError
from numba.tests.support import TestCase
from numba.tests.enum_usecases import Shake, RequestError
from numba.np import numpy_support
def test_layout_checker(self):

    def check_arr(arr):
        dims = arr.shape
        strides = arr.strides
        itemsize = arr.dtype.itemsize
        is_c = numpy_support.is_contiguous(dims, strides, itemsize)
        is_f = numpy_support.is_fortran(dims, strides, itemsize)
        expect_c = arr.flags['C_CONTIGUOUS']
        expect_f = arr.flags['F_CONTIGUOUS']
        self.assertEqual(is_c, expect_c)
        self.assertEqual(is_f, expect_f)
    arr = np.arange(24)
    check_arr(arr)
    check_arr(arr.reshape((3, 8)))
    check_arr(arr.reshape((3, 8)).T)
    check_arr(arr.reshape((3, 8))[::2])
    check_arr(arr.reshape((2, 3, 4)))
    check_arr(arr.reshape((2, 3, 4)).T)
    check_arr(arr.reshape((2, 3, 4))[:, ::3])
    check_arr(arr.reshape((2, 3, 4)).T[:, ::3])
    check_arr(arr.reshape((2, 3, 4))[::2])
    check_arr(arr.reshape((2, 3, 4)).T[:, :, ::2])
    check_arr(arr.reshape((2, 3, 4))[::2, ::3])
    check_arr(arr.reshape((2, 3, 4)).T[:, ::3, ::2])
    check_arr(arr.reshape((2, 3, 4))[::2, ::3, ::4])
    check_arr(arr.reshape((2, 3, 4)).T[::4, ::3, ::2])
    check_arr(arr.reshape((2, 2, 3, 2))[::2, ::2, ::3])
    check_arr(arr.reshape((2, 2, 3, 2)).T[:, ::3, ::2, ::2])
    check_arr(arr.reshape((2, 2, 3, 2))[::5, ::2, ::3])
    check_arr(arr.reshape((2, 2, 3, 2)).T[:, ::3, ::2, ::5])
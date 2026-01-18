import decimal
import itertools
import numpy as np
import unittest
from numba import jit, njit, typeof
from numba.core import utils, types, errors
from numba.tests.support import TestCase, tag
from numba.core.typing import arraydecl
from numba.core.types import intp, ellipsis, slice2_type, slice3_type
def test_ellipsis_issue1498(self):

    @njit
    def udt(arr):
        out = np.zeros_like(arr)
        i = 0
        for index, val in np.ndenumerate(arr[..., i]):
            out[index][i] = val
        return out
    py_func = udt.py_func
    outersize = 4
    innersize = 4
    arr = np.arange(outersize * innersize).reshape(outersize, innersize)
    got = udt(arr)
    expected = py_func(arr)
    np.testing.assert_equal(got, expected)
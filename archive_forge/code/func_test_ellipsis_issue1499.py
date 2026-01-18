import decimal
import itertools
import numpy as np
import unittest
from numba import jit, njit, typeof
from numba.core import utils, types, errors
from numba.tests.support import TestCase, tag
from numba.core.typing import arraydecl
from numba.core.types import intp, ellipsis, slice2_type, slice3_type
def test_ellipsis_issue1499(self):

    @njit
    def udt(arr):
        return arr[..., 0]
    arr = np.arange(3)
    got = udt(arr)
    expected = udt.py_func(arr)
    np.testing.assert_equal(got, expected)
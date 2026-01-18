import decimal
import itertools
import numpy as np
import unittest
from numba import jit, njit, typeof
from numba.core import utils, types, errors
from numba.tests.support import TestCase, tag
from numba.core.typing import arraydecl
from numba.core.types import intp, ellipsis, slice2_type, slice3_type
def test_2d_integer_indexing_via_call(self):

    @njit
    def index1(X, i0):
        return X[i0]

    @njit
    def index2(X, i0, i1):
        return index1(X[i0], i1)
    a = np.arange(10).reshape(2, 5)
    self.assertEqual(index2(a, 0, 0), a[0][0])
    self.assertEqual(index2(a, 1, 1), a[1][1])
    self.assertEqual(index2(a, -1, -1), a[-1][-1])
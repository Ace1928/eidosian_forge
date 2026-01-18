import array
import cmath
from functools import reduce
import itertools
from operator import mul
import math
import symengine as se
from symengine.test_utilities import raises
from symengine import have_numpy
import unittest
from unittest.case import SkipTest
@unittest.skipUnless(have_numpy, 'Numpy not installed')
def test_broadcast_fortran():
    n = 3
    inp = np.arange(2 * n).reshape((n, 2), order='F')
    lmb, check = _get_2_to_2by2()
    A = lmb(inp)
    assert A.shape == (3, 2, 2)
    for i in range(n):
        check(A[i, ...], inp[i, :])
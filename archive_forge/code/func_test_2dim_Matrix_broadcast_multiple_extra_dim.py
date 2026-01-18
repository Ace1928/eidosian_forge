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
def test_2dim_Matrix_broadcast_multiple_extra_dim():
    L, check = _get_1_to_2by3_matrix()
    inp = np.arange(1, 4 * 5 * 6 + 1).reshape((4, 5, 6))
    out = L(inp)
    assert out.shape == (4, 5, 6, 2, 3)
    for i, j, k in itertools.product(range(4), range(5), range(6)):
        check(out[i, j, k, ...], (inp[i, j, k],))
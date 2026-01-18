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
def test_more_than_255_args():
    n = 257
    x = se.symarray('x', n)
    p, q, r = (17, 42, 13)
    terms = [i * s for i, s in enumerate(x, p)]
    exprs = [se.add(*terms), r + x[0], -99]
    callback = se.Lambdify(x, exprs)
    input_arr = np.arange(q, q + n * n).reshape((n, n))
    out = callback(input_arr)
    ref = np.empty((n, 3))
    coeffs = np.arange(p, p + n, dtype=np.int64)
    for i in range(n):
        ref[i, 0] = coeffs.dot(np.arange(q + n * i, q + n * (i + 1), dtype=np.int64))
        ref[i, 1] = q + n * i + r
    ref[:, 2] = -99
    assert np.allclose(out, ref)
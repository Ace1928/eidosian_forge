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
def test_broadcast_multiple_extra_dimensions():
    inp = np.arange(12.0).reshape((4, 3, 1))
    x = se.symbols('x')
    cb = se.Lambdify([x], [x ** 2, x ** 3])
    assert np.allclose(cb([inp[0, 2]]), [4, 8])
    out = cb(inp)
    assert out.shape == (4, 3, 1, 2)
    out = out.squeeze()
    assert abs(out[2, 1, 0] - 7 ** 2) < 1e-14
    assert abs(out[2, 1, 1] - 7 ** 3) < 1e-14
    assert abs(out[-1, -1, 0] - 11 ** 2) < 1e-14
    assert abs(out[-1, -1, 1] - 11 ** 3) < 1e-14
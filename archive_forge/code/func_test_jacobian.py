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
def test_jacobian():
    x, y = se.symbols('x, y')
    args = se.DenseMatrix(2, 1, [x, y])
    v = se.DenseMatrix(2, 1, [x ** 3 * y, (x + 1) * (y + 1)])
    jac = v.jacobian(args)
    lmb = se.Lambdify(args, jac)
    out = np.empty((2, 2))
    inp = X, Y = (7, 11)
    lmb(inp, out=out)
    assert np.allclose(out, [[3 * X ** 2 * Y, X ** 3], [Y + 1, X + 1]])
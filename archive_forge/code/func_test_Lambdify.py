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
def test_Lambdify():
    n = 7
    args = x, y, z = se.symbols('x y z')
    L = se.Lambdify(args, [x + y + z, x ** 2, (x - y) / z, x * y * z], backend='lambda')
    assert allclose(L(range(n, n + len(args))), [3 * n + 3, n ** 2, -1 / (n + 2), n * (n + 1) * (n + 2)])
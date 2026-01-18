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
def test_complex_2():
    x = se.Symbol('x')
    lmb = se.Lambdify([x], [3 + x - 1j], real=False)
    assert abs(lmb([11 + 13j])[0] - (14 + 12j)) < 1e-15
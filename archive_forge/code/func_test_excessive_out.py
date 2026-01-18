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
def test_excessive_out():
    x = se.symbols('x')
    lmb = se.Lambdify([x], [-x])
    inp = np.ones(1)
    out = np.ones(2)
    _ = lmb(inp, out=out[:inp.size])
    assert np.allclose(inp, [1, 1])
    assert out[0] == -1
    assert out[1] == 1
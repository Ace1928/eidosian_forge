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
def test_cse_big():
    args, exprs, inp = _get_cse_exprs_big()
    lmb = se.Lambdify(args, exprs, cse=True)
    out = lmb(inp)
    ref = [expr.xreplace(dict(zip(args, inp))) for expr in exprs]
    assert allclose(out, ref)
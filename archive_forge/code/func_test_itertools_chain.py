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
def test_itertools_chain():
    args, exprs, inp, check = _get_array()
    L = se.Lambdify(args, exprs)
    inp = itertools.chain([inp[0]], (inp[1],), [inp[2]])
    A = L(inp)
    check(A)
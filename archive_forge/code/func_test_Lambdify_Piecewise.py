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
def test_Lambdify_Piecewise():
    _test_Lambdify_Piecewise(lambda *args: se.Lambdify(*args, backend='lambda'))
    if se.have_llvm:
        _test_Lambdify_Piecewise(lambda *args: se.Lambdify(*args, backend='llvm'))
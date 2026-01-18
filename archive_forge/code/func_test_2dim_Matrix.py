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
def test_2dim_Matrix():
    L, check = _get_1_to_2by3_matrix()
    inp = [7]
    check(L(inp), inp)
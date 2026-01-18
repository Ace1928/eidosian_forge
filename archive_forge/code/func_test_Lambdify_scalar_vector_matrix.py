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
def test_Lambdify_scalar_vector_matrix():
    _test_Lambdify_scalar_vector_matrix(lambda *args: se.Lambdify(*args, backend='lambda'))
    if se.have_llvm:
        _test_Lambdify_scalar_vector_matrix(lambda *args: se.Lambdify(*args, backend='llvm'))
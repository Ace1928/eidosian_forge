import itertools
import math
import sys
from numba import jit, types
from numba.tests.support import TestCase
from .complex_usecases import *
import unittest
def test_atan_npm(self):
    self.check_unary_func(atan_usecase, no_pyobj_flags, ulps=2)
import itertools
import math
import sys
from numba import jit, types
from numba.tests.support import TestCase
from .complex_usecases import *
import unittest
def test_isfinite_npm(self):
    self.check_predicate_func(isfinite_usecase, no_pyobj_flags)
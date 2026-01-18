import unittest
import math
import sys
from numba import jit
from numba.core import utils
from numba.tests.support import TestCase, tag
def test_constant_uint64_function_call(self, nopython=False):
    pyfunc = usecase_uint64_func
    self.check_nullary_func(pyfunc, nopython=nopython)
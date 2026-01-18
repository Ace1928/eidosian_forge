import unittest
import math
import sys
from numba import jit
from numba.core import utils
from numba.tests.support import TestCase, tag
def test_constant_uint64_npm(self):
    self.test_constant_uint64(nopython=True)
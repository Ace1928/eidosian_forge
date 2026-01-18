import unittest
import math
import sys
from numba import jit
from numba.core import utils
from numba.tests.support import TestCase, tag
def check_nullary_func(self, pyfunc, **kwargs):
    cfunc = jit(**kwargs)(pyfunc)
    self.assertPreciseEqual(cfunc(), pyfunc())
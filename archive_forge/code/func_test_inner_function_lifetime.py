import gc
import weakref
from numba import jit
from numba.core import types
from numba.tests.support import TestCase
import unittest
def test_inner_function_lifetime(self):
    self.check_inner_function_lifetime(forceobj=True)
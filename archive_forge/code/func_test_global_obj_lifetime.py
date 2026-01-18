import gc
import weakref
from numba import jit
from numba.core import types
from numba.tests.support import TestCase
import unittest
def test_global_obj_lifetime(self):
    self.check_global_obj_lifetime(forceobj=True)
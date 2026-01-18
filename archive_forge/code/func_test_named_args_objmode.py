from numba import int32, int64
from numba import jit
from numba.core import types
from numba.extending import overload
from numba.tests.support import TestCase, tag
import unittest
def test_named_args_objmode(self):
    self.test_named_args(objmode=True)
import unittest
import dis
import struct
from numba import jit
from numba.core import utils
from numba.tests.support import TestCase
def test_extended_arg_load_const(self):
    pyfunc = self.get_extended_arg_load_const()
    self.assertGreater(len(pyfunc.__code__.co_consts), self.bytecode_len)
    self.assertPreciseEqual(pyfunc(), 42)
    cfunc = jit(nopython=True)(pyfunc)
    self.assertPreciseEqual(cfunc(), 42)
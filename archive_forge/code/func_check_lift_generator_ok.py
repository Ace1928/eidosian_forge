from io import StringIO
import numpy as np
from numba.core import types
from numba.core.compiler import compile_extra, Flags
from numba.tests.support import TestCase, tag, MemoryLeakMixin
import unittest
def check_lift_generator_ok(self, pyfunc, argtypes, args):
    """
        Check that pyfunc (a generator function) can loop-lift even in
        nopython mode.
        """
    cres = self.try_lift(pyfunc, argtypes)
    expected = list(pyfunc(*args))
    got = list(cres.entry_point(*args))
    self.assert_lifted_native(cres)
    self.assertPreciseEqual(expected, got)
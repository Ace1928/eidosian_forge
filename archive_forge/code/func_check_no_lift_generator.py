from io import StringIO
import numpy as np
from numba.core import types
from numba.core.compiler import compile_extra, Flags
from numba.tests.support import TestCase, tag, MemoryLeakMixin
import unittest
def check_no_lift_generator(self, pyfunc, argtypes, args):
    """
        Check that pyfunc (a generator function) can't loop-lift.
        """
    cres = compile_isolated(pyfunc, argtypes, flags=looplift_flags)
    self.assertFalse(cres.lifted)
    expected = list(pyfunc(*args))
    got = list(cres.entry_point(*args))
    self.assertPreciseEqual(expected, got)
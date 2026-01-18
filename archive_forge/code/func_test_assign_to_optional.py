import itertools
import numpy as np
import unittest
from numba import jit, njit
from numba.core import types
from numba.tests.support import TestCase
def test_assign_to_optional(self):
    """
        Check that we can assign to a variable of optional type
        """

    @njit
    def make_optional(val, get_none):
        if get_none:
            ret = None
        else:
            ret = val
        return ret

    @njit
    def foo(val, run_second):
        a = make_optional(val, True)
        if run_second:
            a = make_optional(val, False)
        return a
    self.assertIsNone(foo(123, False))
    self.assertEqual(foo(231, True), 231)
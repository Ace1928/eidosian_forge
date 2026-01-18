import gc
import itertools
import numpy as np
import unittest
from numba import jit, njit
from numba.core import types
from numba.tests.support import TestCase
from numba.np import numpy_support
def test_multiple_args_negative_to_unsigned(self):
    pyfunc = foobar
    cfunc = njit(types.uint64(types.uint64, types.uint64, types.uint64))(pyfunc)
    test_fail_args = ((-1, 0, 1), (0, -1, 1), (0, 1, -1))
    with self.assertRaises(OverflowError):
        for a, b, c in test_fail_args:
            cfunc(a, b, c)
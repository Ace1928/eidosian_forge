import gc
import itertools
import numpy as np
import unittest
from numba import jit, njit
from numba.core import types
from numba.tests.support import TestCase
from numba.np import numpy_support
def test_multiple_args_records(self):
    pyfunc = foobar
    mystruct_dt = np.dtype([('p', np.float64), ('row', np.float64), ('col', np.float64)])
    mystruct = numpy_support.from_dtype(mystruct_dt)
    cfunc = njit(mystruct[:](mystruct[:], types.uint64, types.uint64))(pyfunc)
    st1 = np.recarray(3, dtype=mystruct_dt)
    st1.p = np.arange(st1.size) + 1
    st1.row = np.arange(st1.size) + 1
    st1.col = np.arange(st1.size) + 1
    with self.assertRefCount(st1):
        test_fail_args = ((st1, -1, 1), (st1, 1, -1))
        for a, b, c in test_fail_args:
            with self.assertRaises(OverflowError):
                cfunc(a, b, c)
        del test_fail_args, a, b, c
        gc.collect()
import itertools
import math
import sys
import unittest
import warnings
import numpy as np
from numba import njit, types
from numba.tests.support import TestCase
from numba.np import numpy_support
def test_hypot(self):
    pyfunc = hypot
    x_types = [types.int64, types.uint64, types.float32, types.float64]
    x_values = [1, 2, 3, 4, 5, 6, 0.21, 0.34]
    y_values = [x + 2 for x in x_values]
    prec = 'single'
    self.run_binary(pyfunc, x_types, x_values, y_values, prec)

    def naive_hypot(x, y):
        return math.sqrt(x * x + y * y)
    cfunc = njit(pyfunc)
    for fltty in (types.float32, types.float64):
        dt = numpy_support.as_dtype(fltty).type
        val = dt(np.finfo(dt).max / 30.0)
        nb_ans = cfunc(val, val)
        self.assertPreciseEqual(nb_ans, pyfunc(val, val), prec='single')
        self.assertTrue(np.isfinite(nb_ans))
        with warnings.catch_warnings():
            warnings.simplefilter('error', RuntimeWarning)
            self.assertRaisesRegex(RuntimeWarning, 'overflow encountered in .*scalar', naive_hypot, val, val)
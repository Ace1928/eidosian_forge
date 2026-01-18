import itertools
import functools
import sys
import operator
from collections import namedtuple
import numpy as np
import unittest
import warnings
from numba import jit, typeof, njit, typed
from numba.core import errors, types, config
from numba.tests.support import (TestCase, tag, ignore_internal_warnings,
from numba.core.extending import overload_method, box
def test_divmod_floats(self, flags=forceobj_flags):
    pyfunc = divmod_usecase
    cfunc = jit((types.float64, types.float64), **flags)(pyfunc)
    denominators = [1.0, 3.5, 1e+100, -2.0, -7.5, -1e+101, np.inf, -np.inf, np.nan]
    numerators = denominators + [-0.0, 0.0]
    for x, y in itertools.product(numerators, denominators):
        expected_quot, expected_rem = pyfunc(x, y)
        quot, rem = cfunc(x, y)
        self.assertPreciseEqual((quot, rem), (expected_quot, expected_rem))
    for x in numerators:
        with self.assertRaises(ZeroDivisionError):
            cfunc(x, 0.0)
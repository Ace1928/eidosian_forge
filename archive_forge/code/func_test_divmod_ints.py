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
def test_divmod_ints(self, flags=forceobj_flags):
    pyfunc = divmod_usecase
    cfunc = jit((types.int64, types.int64), **flags)(pyfunc)

    def truncate_result(x, bits=64):
        if x >= 0:
            x &= (1 << bits - 1) - 1
        return x
    denominators = [1, 3, 7, 15, -1, -3, -7, -15, 2 ** 63 - 1, -2 ** 63]
    numerators = denominators + [0]
    for x, y in itertools.product(numerators, denominators):
        expected_quot, expected_rem = pyfunc(x, y)
        quot, rem = cfunc(x, y)
        f = truncate_result
        self.assertPreciseEqual((f(quot), f(rem)), (f(expected_quot), f(expected_rem)))
    for x in numerators:
        with self.assertRaises(ZeroDivisionError):
            cfunc(x, 0)
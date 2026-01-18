from itertools import product, cycle
import gc
import sys
import unittest
import warnings
import numpy as np
from numba import jit, njit, typeof
from numba.core import types
from numba.core.errors import TypingError, NumbaValueError
from numba.np.numpy_support import as_dtype, numpy_version
from numba.tests.support import TestCase, MemoryLeakMixin, needs_blas
def check_round_scalar(self, unary_pyfunc, binary_pyfunc):
    base_values = [-3.0, -2.5, -2.25, -1.5, 1.5, 2.25, 2.5, 2.75]
    complex_values = [x * (1 - 1j) for x in base_values]
    int_values = [int(x) for x in base_values]
    argtypes = (types.float64, types.float32, types.int32, types.complex64, types.complex128)
    argvalues = [base_values, base_values, int_values, complex_values, complex_values]
    pyfunc = binary_pyfunc
    for ty, values in zip(argtypes, argvalues):
        cfunc = njit((ty, types.int32))(pyfunc)
        for decimals in (1, 0, -1):
            for v in values:
                if decimals > 0:
                    v *= 10
                expected = _fixed_np_round(v, decimals)
                got = cfunc(v, decimals)
                self.assertPreciseEqual(got, expected)
    pyfunc = unary_pyfunc
    for ty, values in zip(argtypes, argvalues):
        cfunc = njit((ty,))(pyfunc)
        for v in values:
            expected = _fixed_np_round(v)
            got = cfunc(v)
            self.assertPreciseEqual(got, expected)
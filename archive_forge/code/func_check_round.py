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
def check_round(cfunc, values, inty, outty, decimals):
    arr = values.astype(as_dtype(inty))
    out = np.zeros_like(arr).astype(as_dtype(outty))
    pyout = out.copy()
    _fixed_np_round(arr, decimals, pyout)
    self.memory_leak_setup()
    cfunc(arr, decimals, out)
    self.memory_leak_teardown()
    np.testing.assert_allclose(out, pyout)
    with self.assertRaises(ValueError) as raises:
        cfunc(arr, decimals, out[1:])
    self.assertEqual(str(raises.exception), 'invalid output shape')
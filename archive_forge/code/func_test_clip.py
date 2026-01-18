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
def test_clip(self):
    has_out = (np_clip, np_clip_kwargs, array_clip, array_clip_kwargs)
    has_no_out = (np_clip_no_out, array_clip_no_out)
    for a in (np.linspace(-10, 10, 101), np.linspace(-10, 10, 40).reshape(5, 2, 4)):
        for pyfunc in has_out + has_no_out:
            cfunc = jit(nopython=True)(pyfunc)
            msg = 'array_clip: must set either max or min'
            with self.assertRaisesRegex(ValueError, msg):
                cfunc(a, None, None)
            np.testing.assert_equal(pyfunc(a, 0, None), cfunc(a, 0, None))
            np.testing.assert_equal(pyfunc(a, None, 0), cfunc(a, None, 0))
            np.testing.assert_equal(pyfunc(a, -5, 5), cfunc(a, -5, 5))
            if pyfunc in has_out:
                pyout = np.empty_like(a)
                cout = np.empty_like(a)
                np.testing.assert_equal(pyfunc(a, -5, 5, pyout), cfunc(a, -5, 5, cout))
                np.testing.assert_equal(pyout, cout)
            self._lower_clip_result_test_util(cfunc, a, -5, 5)
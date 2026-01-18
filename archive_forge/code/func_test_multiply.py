import functools
import itertools
import sys
import warnings
import threading
import operator
import numpy as np
import unittest
from numba import guvectorize, njit, typeof, vectorize
from numba.core import types
from numba.np.numpy_support import from_dtype
from numba.core.errors import LoweringError, TypingError
from numba.tests.support import TestCase, MemoryLeakMixin
from numba.core.typing.npydecl import supported_ufuncs
from numba.np import numpy_support
from numba.core.registry import cpu_target
from numba.core.base import BaseContext
from numba.np import ufunc_db
def test_multiply(self):
    ufunc = np.multiply
    fn = _make_ufunc_usecase(ufunc)
    self._check_ufunc_with_dtypes(fn, ufunc, ['m8[s]', 'q', 'm8[us]'])
    self._check_ufunc_with_dtypes(fn, ufunc, ['q', 'm8[s]', 'm8[us]'])
    with self.assertRaises(LoweringError):
        self._check_ufunc_with_dtypes(fn, ufunc, ['m8[s]', 'q', 'm8[m]'])
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
def test_floor_divide_array_op(self):
    self.inputs = [(np.uint32(1), types.uint32), (np.int32(-2), types.int32), (np.int32(0), types.int32), (np.uint64(4), types.uint64), (np.int64(-5), types.int64), (np.int64(0), types.int64), (np.float32(-0.5), types.float32), (np.float32(1.5), types.float32), (np.float64(-2.5), types.float64), (np.float64(3.5), types.float64), (np.array([1, 2], dtype='u4'), types.Array(types.uint32, 1, 'C')), (np.array([3, 4], dtype='u8'), types.Array(types.uint64, 1, 'C')), (np.array([-1, 1, 5], dtype='i4'), types.Array(types.int32, 1, 'C')), (np.array([-1, 1, 6], dtype='i8'), types.Array(types.int64, 1, 'C')), (np.array([-0.5, 1.5], dtype='f4'), types.Array(types.float32, 1, 'C')), (np.array([-2.5, 3.5], dtype='f8'), types.Array(types.float64, 1, 'C'))]
    self.binary_op_test('//')
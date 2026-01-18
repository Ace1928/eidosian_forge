import random
import numpy as np
from numba.tests.support import TestCase, captured_stdout
from numba import njit, literally
from numba.core import types
from numba.cpython.unsafe.tuple import tuple_setitem, build_full_slice_tuple
from numba.np.unsafe.ndarray import to_fixed_tuple, empty_inferred
from numba.core.unsafe.bytes import memcpy_region
from numba.core.unsafe.refcount import dump_refcount
from numba.cpython.unsafe.numbers import trailing_zeros, leading_zeros
from numba.core.errors import TypingError
def test_memcpy_region(self):

    @njit
    def foo(dst, dst_index, src, src_index, nbytes):
        memcpy_region(dst.ctypes.data, dst_index, src.ctypes.data, src_index, nbytes, 1)
    d = np.zeros(10, dtype=np.int8)
    s = np.arange(10, dtype=np.int8)
    foo(d, 4, s, 1, 5)
    expected = [0, 0, 0, 0, 1, 2, 3, 4, 5, 0]
    np.testing.assert_array_equal(d, expected)
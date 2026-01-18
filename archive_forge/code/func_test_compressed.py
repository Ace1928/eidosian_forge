from os import path
import warnings
import numpy as np
from numpy.testing import (assert_equal, assert_array_equal,
import pytest
from scipy.io import readsav
from scipy.io import _idl
def test_compressed(self):
    s = readsav(path.join(DATA_PATH, 'various_compressed.sav'), verbose=False)
    assert_identical(s.i8u, np.uint8(234))
    assert_identical(s.f32, np.float32(-3.1234567e+37))
    assert_identical(s.c64, np.complex128(1.1987253647623157e+112 - 5.198725888772916e+307j))
    assert_equal(s.array5d.shape, (4, 3, 4, 6, 5))
    assert_identical(s.arrays.a[0], np.array([1, 2, 3], dtype=np.int16))
    assert_identical(s.arrays.b[0], np.array([4.0, 5.0, 6.0, 7.0], dtype=np.float32))
    assert_identical(s.arrays.c[0], np.array([np.complex64(1 + 2j), np.complex64(7 + 8j)]))
    assert_identical(s.arrays.d[0], np.array([b'cheese', b'bacon', b'spam'], dtype=object))
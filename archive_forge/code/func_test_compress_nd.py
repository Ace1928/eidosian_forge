import warnings
import itertools
import pytest
import numpy as np
from numpy.core.numeric import normalize_axis_tuple
from numpy.testing import (
from numpy.ma.testutils import (
from numpy.ma.core import (
from numpy.ma.extras import (
def test_compress_nd(self):
    x = np.array(list(range(3 * 4 * 5))).reshape(3, 4, 5)
    m = np.zeros((3, 4, 5)).astype(bool)
    m[1, 1, 1] = True
    x = array(x, mask=m)
    a = compress_nd(x)
    assert_equal(a, [[[0, 2, 3, 4], [10, 12, 13, 14], [15, 17, 18, 19]], [[40, 42, 43, 44], [50, 52, 53, 54], [55, 57, 58, 59]]])
    a = compress_nd(x, 0)
    assert_equal(a, [[[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14], [15, 16, 17, 18, 19]], [[40, 41, 42, 43, 44], [45, 46, 47, 48, 49], [50, 51, 52, 53, 54], [55, 56, 57, 58, 59]]])
    a = compress_nd(x, 1)
    assert_equal(a, [[[0, 1, 2, 3, 4], [10, 11, 12, 13, 14], [15, 16, 17, 18, 19]], [[20, 21, 22, 23, 24], [30, 31, 32, 33, 34], [35, 36, 37, 38, 39]], [[40, 41, 42, 43, 44], [50, 51, 52, 53, 54], [55, 56, 57, 58, 59]]])
    a2 = compress_nd(x, (1,))
    a3 = compress_nd(x, -2)
    a4 = compress_nd(x, (-2,))
    assert_equal(a, a2)
    assert_equal(a, a3)
    assert_equal(a, a4)
    a = compress_nd(x, 2)
    assert_equal(a, [[[0, 2, 3, 4], [5, 7, 8, 9], [10, 12, 13, 14], [15, 17, 18, 19]], [[20, 22, 23, 24], [25, 27, 28, 29], [30, 32, 33, 34], [35, 37, 38, 39]], [[40, 42, 43, 44], [45, 47, 48, 49], [50, 52, 53, 54], [55, 57, 58, 59]]])
    a2 = compress_nd(x, (2,))
    a3 = compress_nd(x, -1)
    a4 = compress_nd(x, (-1,))
    assert_equal(a, a2)
    assert_equal(a, a3)
    assert_equal(a, a4)
    a = compress_nd(x, (0, 1))
    assert_equal(a, [[[0, 1, 2, 3, 4], [10, 11, 12, 13, 14], [15, 16, 17, 18, 19]], [[40, 41, 42, 43, 44], [50, 51, 52, 53, 54], [55, 56, 57, 58, 59]]])
    a2 = compress_nd(x, (0, -2))
    assert_equal(a, a2)
    a = compress_nd(x, (1, 2))
    assert_equal(a, [[[0, 2, 3, 4], [10, 12, 13, 14], [15, 17, 18, 19]], [[20, 22, 23, 24], [30, 32, 33, 34], [35, 37, 38, 39]], [[40, 42, 43, 44], [50, 52, 53, 54], [55, 57, 58, 59]]])
    a2 = compress_nd(x, (-2, 2))
    a3 = compress_nd(x, (1, -1))
    a4 = compress_nd(x, (-2, -1))
    assert_equal(a, a2)
    assert_equal(a, a3)
    assert_equal(a, a4)
    a = compress_nd(x, (0, 2))
    assert_equal(a, [[[0, 2, 3, 4], [5, 7, 8, 9], [10, 12, 13, 14], [15, 17, 18, 19]], [[40, 42, 43, 44], [45, 47, 48, 49], [50, 52, 53, 54], [55, 57, 58, 59]]])
    a2 = compress_nd(x, (0, -1))
    assert_equal(a, a2)
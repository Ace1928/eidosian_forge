import numpy as np
from numpy.testing import (assert_array_equal, assert_equal,
from numpy.lib.arraysetops import (
import pytest
@pytest.mark.parametrize('ary,prepend,append,expected', [(np.array([1, 2, 3], dtype=np.int16), 2 ** 16, 2 ** 16 + 4, np.array([0, 1, 1, 4], dtype=np.int16)), (np.array([1, 2, 3], dtype=np.float32), np.array([5], dtype=np.float64), None, np.array([5, 1, 1], dtype=np.float32)), (np.array([1, 2, 3], dtype=np.int32), 0, 0, np.array([0, 1, 1, 0], dtype=np.int32)), (np.array([1, 2, 3], dtype=np.int64), 3, -9, np.array([3, 1, 1, -9], dtype=np.int64))])
def test_ediff1d_scalar_handling(self, ary, prepend, append, expected):
    actual = np.ediff1d(ary=ary, to_end=append, to_begin=prepend)
    assert_equal(actual, expected)
    assert actual.dtype == expected.dtype
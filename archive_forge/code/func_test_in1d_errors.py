import numpy as np
from numpy.testing import (assert_array_equal, assert_equal,
from numpy.lib.arraysetops import (
import pytest
def test_in1d_errors(self):
    """Test that in1d raises expected errors."""
    ar1 = np.array([1, 2, 3, 4, 5])
    ar2 = np.array([2, 4, 6, 8, 10])
    assert_raises(ValueError, in1d, ar1, ar2, kind='quicksort')
    obj_ar1 = np.array([1, 'a', 3, 'b', 5], dtype=object)
    obj_ar2 = np.array([1, 'a', 3, 'b', 5], dtype=object)
    assert_raises(ValueError, in1d, obj_ar1, obj_ar2, kind='table')
    for dtype in [np.int32, np.int64]:
        ar1 = np.array([-1, 2, 3, 4, 5], dtype=dtype)
        overflow_ar2 = np.array([-1, np.iinfo(dtype).max], dtype=dtype)
        assert_raises(RuntimeError, in1d, ar1, overflow_ar2, kind='table')
        result = np.in1d(ar1, overflow_ar2, kind=None)
        assert_array_equal(result, [True] + [False] * 4)
        result = np.in1d(ar1, overflow_ar2, kind='sort')
        assert_array_equal(result, [True] + [False] * 4)
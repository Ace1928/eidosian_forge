import pytest
import numpy as np
from numpy.testing import (
from numpy.lib.index_tricks import (
def test_big_indices(self):
    if np.intp == np.int64:
        arr = ([1, 29], [3, 5], [3, 117], [19, 2], [2379, 1284], [2, 2], [0, 1])
        assert_equal(np.ravel_multi_index(arr, (41, 7, 120, 36, 2706, 8, 6)), [5627771580, 117259570957])
    assert_raises(ValueError, np.unravel_index, 1, (2 ** 32 - 1, 2 ** 31 + 1))
    dummy_arr = ([0], [0])
    half_max = np.iinfo(np.intp).max // 2
    assert_equal(np.ravel_multi_index(dummy_arr, (half_max, 2)), [0])
    assert_raises(ValueError, np.ravel_multi_index, dummy_arr, (half_max + 1, 2))
    assert_equal(np.ravel_multi_index(dummy_arr, (half_max, 2), order='F'), [0])
    assert_raises(ValueError, np.ravel_multi_index, dummy_arr, (half_max + 1, 2), order='F')
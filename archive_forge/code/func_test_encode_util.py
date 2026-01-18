import pickle
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from sklearn.utils._encode import _check_unknown, _encode, _get_counts, _unique
@pytest.mark.parametrize('values, expected', [(np.array([2, 1, 3, 1, 3], dtype='int64'), np.array([1, 2, 3], dtype='int64')), (np.array([2, 1, np.nan, 1, np.nan], dtype='float32'), np.array([1, 2, np.nan], dtype='float32')), (np.array(['b', 'a', 'c', 'a', 'c'], dtype=object), np.array(['a', 'b', 'c'], dtype=object)), (np.array(['b', 'a', None, 'a', None], dtype=object), np.array(['a', 'b', None], dtype=object)), (np.array(['b', 'a', 'c', 'a', 'c']), np.array(['a', 'b', 'c']))], ids=['int64', 'float32-nan', 'object', 'object-None', 'str'])
def test_encode_util(values, expected):
    uniques = _unique(values)
    assert_array_equal(uniques, expected)
    result, encoded = _unique(values, return_inverse=True)
    assert_array_equal(result, expected)
    assert_array_equal(encoded, np.array([1, 0, 2, 0, 2]))
    encoded = _encode(values, uniques=uniques)
    assert_array_equal(encoded, np.array([1, 0, 2, 0, 2]))
    result, counts = _unique(values, return_counts=True)
    assert_array_equal(result, expected)
    assert_array_equal(counts, np.array([2, 1, 2]))
    result, encoded, counts = _unique(values, return_inverse=True, return_counts=True)
    assert_array_equal(result, expected)
    assert_array_equal(encoded, np.array([1, 0, 2, 0, 2]))
    assert_array_equal(counts, np.array([2, 1, 2]))
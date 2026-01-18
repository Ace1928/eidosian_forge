import numpy
import pytest
import modin.numpy as np
from .utils import assert_scalar_or_array_equal
def test_all_any_where():
    arr = np.array([[0, 1], [1, 0]])
    where = np.array([[False, True], [True, False]])
    result = arr.all(where=where)
    assert result
    where = np.array([[True, False], [False, False]])
    result = arr.all(where=where, axis=1)
    assert_scalar_or_array_equal(result, numpy.array([False, True]))
    result = arr.all(where=False, axis=1)
    assert_scalar_or_array_equal(result, numpy.array([True, True]))
    result = arr.all(where=False, axis=0)
    assert_scalar_or_array_equal(result, numpy.array([True, True]))
    assert bool(arr.all(where=False, axis=None))
    where = np.array([[True, False], [False, True]])
    result = arr.any(where=where)
    assert not result
    where = np.array([[False, True], [False, False]])
    result = arr.any(where=where, axis=1)
    assert_scalar_or_array_equal(result, numpy.array([True, False]))
    result = arr.any(where=False, axis=1)
    assert_scalar_or_array_equal(result, numpy.array([False, False]))
    result = arr.any(where=False, axis=0)
    assert_scalar_or_array_equal(result, numpy.array([False, False]))
    assert not bool(arr.any(where=False, axis=None))
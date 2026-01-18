import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose, assert_equal
from numpy.lib.arraypad import _as_pairs
@pytest.mark.parametrize('dtype', _numeric_dtypes)
def test_negative_difference(self, dtype):
    """
        Check correct behavior of unsigned dtypes if there is a negative
        difference between the edge to pad and `end_values`. Check both cases
        to be independent of implementation. Test behavior for all other dtypes
        in case dtype casting interferes with complex dtypes. See gh-14191.
        """
    x = np.array([3], dtype=dtype)
    result = np.pad(x, 3, mode='linear_ramp', end_values=0)
    expected = np.array([0, 1, 2, 3, 2, 1, 0], dtype=dtype)
    assert_equal(result, expected)
    x = np.array([0], dtype=dtype)
    result = np.pad(x, 3, mode='linear_ramp', end_values=3)
    expected = np.array([3, 2, 1, 0, 1, 2, 3], dtype=dtype)
    assert_equal(result, expected)
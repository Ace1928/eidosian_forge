import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('a,b', [(np.array([1, 2, 3]), [1, 2, 3]), (np.array([1, 2, 3]), np.array([1.0, 2.0, 3.0])), (iter([1, 2, 3]), [1, 2, 3]), ([1, 2, 3], [1, 2, 4]), ([1, 2, 3], [1, 2, 3, 4]), ([1, 2, 3], 1)])
def test_assert_not_almost_equal_iterables(a, b):
    _assert_not_almost_equal(a, b)
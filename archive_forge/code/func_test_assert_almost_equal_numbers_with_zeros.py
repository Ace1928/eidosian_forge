import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('a,b', [(0, 0), (0, 0.0), (0, np.float64(0)), (1e-08, 0)])
def test_assert_almost_equal_numbers_with_zeros(a, b):
    _assert_almost_equal_both(a, b)
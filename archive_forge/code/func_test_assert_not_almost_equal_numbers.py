import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('a,b', [(1.1, 1), (1.1, True), (1, 2), (1.0001, np.int16(1)), (0.1, 0.1001), (0.0011, 0.0012)])
def test_assert_not_almost_equal_numbers(a, b):
    _assert_not_almost_equal_both(a, b)
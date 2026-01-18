import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('a,b', [(1.1, 1.1), (1.1, 1.100001), (1.1, 1.1001), (1000.0, 1000.0005), (1.1, 1.11), (0.1, 0.101)])
def test_assert_almost_equal_numbers_rtol(a, b):
    _assert_almost_equal_both(a, b, rtol=0.05)
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('a,b', [(1.1, 1.1), (1.1, 1.100001), (1.1, 1.1001), (1e-06, 5e-06), (1000.0, 1000.0005), (1.1e-05, 1.2e-05)])
def test_assert_almost_equal_numbers_atol(a, b):
    _assert_almost_equal_both(a, b, rtol=0.0005, atol=0.0005)
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('method', ['skew', 'kurt'])
def test_expanding_skew_kurt_numerical_stability(method):
    s = Series(np.random.default_rng(2).random(10))
    expected = getattr(s.expanding(3), method)()
    s = s + 5000
    result = getattr(s.expanding(3), method)()
    tm.assert_series_equal(result, expected)
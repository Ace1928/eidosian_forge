import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_ewm_consistency_mean(all_data, adjust, ignore_na, min_periods):
    com = 3.0
    result = all_data.ewm(com=com, min_periods=min_periods, adjust=adjust, ignore_na=ignore_na).mean()
    weights = create_mock_weights(all_data, com=com, adjust=adjust, ignore_na=ignore_na)
    expected = all_data.multiply(weights).cumsum().divide(weights.cumsum()).ffill()
    expected[all_data.expanding().count() < (max(min_periods, 1) if min_periods else 1)] = np.nan
    tm.assert_equal(result, expected.astype('float64'))
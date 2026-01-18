import numpy as np
import pytest
from pandas import Series
import pandas._testing as tm
def test_expanding_consistency_constant(consistent_data, min_periods):
    count_x = consistent_data.expanding().count()
    mean_x = consistent_data.expanding(min_periods=min_periods).mean()
    corr_x_x = consistent_data.expanding(min_periods=min_periods).corr(consistent_data)
    exp = consistent_data.max() if isinstance(consistent_data, Series) else consistent_data.max().max()
    expected = consistent_data * np.nan
    expected[count_x >= max(min_periods, 1)] = exp
    tm.assert_equal(mean_x, expected)
    expected[:] = np.nan
    tm.assert_equal(corr_x_x, expected)
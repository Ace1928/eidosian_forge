import numpy as np
import pytest
from pandas import Series
import pandas._testing as tm
@pytest.mark.parametrize('ddof', [0, 1])
def test_expanding_consistency_series_cov_corr(series_data, min_periods, ddof):
    var_x_plus_y = (series_data + series_data).expanding(min_periods=min_periods).var(ddof=ddof)
    var_x = series_data.expanding(min_periods=min_periods).var(ddof=ddof)
    var_y = series_data.expanding(min_periods=min_periods).var(ddof=ddof)
    cov_x_y = series_data.expanding(min_periods=min_periods).cov(series_data, ddof=ddof)
    tm.assert_equal(cov_x_y, 0.5 * (var_x_plus_y - var_x - var_y))
    corr_x_y = series_data.expanding(min_periods=min_periods).corr(series_data)
    std_x = series_data.expanding(min_periods=min_periods).std(ddof=ddof)
    std_y = series_data.expanding(min_periods=min_periods).std(ddof=ddof)
    tm.assert_equal(corr_x_y, cov_x_y / (std_x * std_y))
    if ddof == 0:
        mean_x = series_data.expanding(min_periods=min_periods).mean()
        mean_y = series_data.expanding(min_periods=min_periods).mean()
        mean_x_times_y = (series_data * series_data).expanding(min_periods=min_periods).mean()
        tm.assert_equal(cov_x_y, mean_x_times_y - mean_x * mean_y)
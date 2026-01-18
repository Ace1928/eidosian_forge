import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('min_periods', [0, 1, 2])
@pytest.mark.parametrize('name', ['cov', 'corr'])
def test_ewm_corr_cov_min_periods(name, min_periods):
    A = Series(np.random.default_rng(2).standard_normal(50), index=range(50))
    B = A[2:] + np.random.default_rng(2).standard_normal(48)
    A[:10] = np.nan
    B.iloc[-10:] = np.nan
    result = getattr(A.ewm(com=20, min_periods=min_periods), name)(B)
    assert np.isnan(result.values[:11]).all()
    assert not np.isnan(result.values[11:]).any()
    empty = Series([], dtype=np.float64)
    result = getattr(empty.ewm(com=50, min_periods=min_periods), name)(empty)
    tm.assert_series_equal(result, empty)
    result = getattr(Series([1.0]).ewm(com=50, min_periods=min_periods), name)(Series([1.0]))
    tm.assert_series_equal(result, Series([np.nan]))
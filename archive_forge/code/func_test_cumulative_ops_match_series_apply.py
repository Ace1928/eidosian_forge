import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('method', ['cumsum', 'cumprod', 'cummin', 'cummax'])
def test_cumulative_ops_match_series_apply(self, datetime_frame, method):
    datetime_frame.iloc[5:10, 0] = np.nan
    datetime_frame.iloc[10:15, 1] = np.nan
    datetime_frame.iloc[15:, 2] = np.nan
    result = getattr(datetime_frame, method)()
    expected = datetime_frame.apply(getattr(Series, method))
    tm.assert_frame_equal(result, expected)
    result = getattr(datetime_frame, method)(axis=1)
    expected = datetime_frame.apply(getattr(Series, method), axis=1)
    tm.assert_frame_equal(result, expected)
    assert np.shape(result) == np.shape(datetime_frame)
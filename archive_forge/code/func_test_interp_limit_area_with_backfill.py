import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('data, expected_data, kwargs', (([np.nan, np.nan, 3, np.nan, np.nan, np.nan, 7, np.nan, np.nan], [np.nan, np.nan, 3.0, 7.0, 7.0, 7.0, 7.0, np.nan, np.nan], {'method': 'bfill', 'limit_area': 'inside'}), ([np.nan, np.nan, 3, np.nan, np.nan, np.nan, 7, np.nan, np.nan], [np.nan, np.nan, 3.0, np.nan, np.nan, 7.0, 7.0, np.nan, np.nan], {'method': 'bfill', 'limit_area': 'inside', 'limit': 1}), ([np.nan, np.nan, 3, np.nan, np.nan, np.nan, 7, np.nan, np.nan], [3.0, 3.0, 3.0, np.nan, np.nan, np.nan, 7.0, np.nan, np.nan], {'method': 'bfill', 'limit_area': 'outside'}), ([np.nan, np.nan, 3, np.nan, np.nan, np.nan, 7, np.nan, np.nan], [np.nan, 3.0, 3.0, np.nan, np.nan, np.nan, 7.0, np.nan, np.nan], {'method': 'bfill', 'limit_area': 'outside', 'limit': 1})))
def test_interp_limit_area_with_backfill(self, data, expected_data, kwargs):
    s = Series(data)
    expected = Series(expected_data)
    msg = 'Series.interpolate with method=bfill'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = s.interpolate(**kwargs)
    tm.assert_series_equal(result, expected)
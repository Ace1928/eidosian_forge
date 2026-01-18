from datetime import datetime
import dateutil.tz
import numpy as np
import pytest
import pytz
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_get_values_for_csv():
    index = pd.date_range(freq='1D', periods=3, start='2017-01-01')
    expected = np.array(['2017-01-01', '2017-01-02', '2017-01-03'], dtype=object)
    result = index._get_values_for_csv()
    tm.assert_numpy_array_equal(result, expected)
    result = index._get_values_for_csv(na_rep='pandas')
    tm.assert_numpy_array_equal(result, expected)
    expected = np.array(['01-2017-01', '01-2017-02', '01-2017-03'], dtype=object)
    result = index._get_values_for_csv(date_format='%m-%Y-%d')
    tm.assert_numpy_array_equal(result, expected)
    index = DatetimeIndex(['2017-01-01', NaT, '2017-01-03'])
    expected = np.array(['2017-01-01', 'NaT', '2017-01-03'], dtype=object)
    result = index._get_values_for_csv(na_rep='NaT')
    tm.assert_numpy_array_equal(result, expected)
    expected = np.array(['2017-01-01', 'pandas', '2017-01-03'], dtype=object)
    result = index._get_values_for_csv(na_rep='pandas')
    tm.assert_numpy_array_equal(result, expected)
    result = index._get_values_for_csv(na_rep='NaT', date_format='%Y-%m-%d %H:%M:%S.%f')
    expected = np.array(['2017-01-01 00:00:00.000000', 'NaT', '2017-01-03 00:00:00.000000'], dtype=object)
    tm.assert_numpy_array_equal(result, expected)
    result = index._get_values_for_csv(na_rep='NaT', date_format='foo')
    expected = np.array(['foo', 'NaT', 'foo'], dtype=object)
    tm.assert_numpy_array_equal(result, expected)
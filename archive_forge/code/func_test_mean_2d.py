import numpy as np
import pytest
from pandas.core.dtypes.dtypes import DatetimeTZDtype
import pandas as pd
from pandas import NaT
import pandas._testing as tm
from pandas.core.arrays import DatetimeArray
def test_mean_2d(self):
    dti = pd.date_range('2016-01-01', periods=6, tz='US/Pacific')
    dta = dti._data.reshape(3, 2)
    result = dta.mean(axis=0)
    expected = dta[1]
    tm.assert_datetime_array_equal(result, expected)
    result = dta.mean(axis=1)
    expected = dta[:, 0] + pd.Timedelta(hours=12)
    tm.assert_datetime_array_equal(result, expected)
    result = dta.mean(axis=None)
    expected = dti.mean()
    assert result == expected
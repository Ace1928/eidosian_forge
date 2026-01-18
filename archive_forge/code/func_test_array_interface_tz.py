from __future__ import annotations
from datetime import timedelta
import operator
import numpy as np
import pytest
from pandas._libs.tslibs import tz_compare
from pandas.core.dtypes.dtypes import DatetimeTZDtype
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import (
def test_array_interface_tz(self):
    tz = 'US/Central'
    data = pd.date_range('2017', periods=2, tz=tz)._data
    result = np.asarray(data)
    expected = np.array([pd.Timestamp('2017-01-01T00:00:00', tz=tz), pd.Timestamp('2017-01-02T00:00:00', tz=tz)], dtype=object)
    tm.assert_numpy_array_equal(result, expected)
    result = np.asarray(data, dtype=object)
    tm.assert_numpy_array_equal(result, expected)
    result = np.asarray(data, dtype='M8[ns]')
    expected = np.array(['2017-01-01T06:00:00', '2017-01-02T06:00:00'], dtype='M8[ns]')
    tm.assert_numpy_array_equal(result, expected)
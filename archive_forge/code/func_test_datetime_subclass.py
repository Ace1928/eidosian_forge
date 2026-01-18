from datetime import (
from dateutil.tz.tz import tzoffset
import numpy as np
import pytest
from pandas._libs import (
from pandas._libs.tslibs.dtypes import NpyDatetimeUnit
from pandas._libs.tslibs.np_datetime import OutOfBoundsDatetime
from pandas import Timestamp
import pandas._testing as tm
@pytest.mark.parametrize('data,expected', [([SubDatetime(2000, 1, 1)], ['2000-01-01T00:00:00.000000000']), ([datetime(2000, 1, 1)], ['2000-01-01T00:00:00.000000000']), ([Timestamp(2000, 1, 1)], ['2000-01-01T00:00:00.000000000'])])
def test_datetime_subclass(data, expected):
    arr = np.array(data, dtype=object)
    result, _ = tslib.array_to_datetime(arr)
    expected = np.array(expected, dtype='M8[ns]')
    tm.assert_numpy_array_equal(result, expected)
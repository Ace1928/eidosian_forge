from datetime import (
from dateutil.tz.tz import tzoffset
import numpy as np
import pytest
from pandas._libs import (
from pandas._libs.tslibs.dtypes import NpyDatetimeUnit
from pandas._libs.tslibs.np_datetime import OutOfBoundsDatetime
from pandas import Timestamp
import pandas._testing as tm
def test_infer_homogeoneous_datetimes_strings(self):
    item = '2023-10-27 18:03:05.678000'
    arr = np.array([None, item, item, item], dtype=object)
    result, tz = tslib.array_to_datetime(arr, creso=creso_infer)
    assert tz is None
    expected = np.array([np.datetime64('NaT'), item, item, item], dtype='M8[us]')
    tm.assert_numpy_array_equal(result, expected)
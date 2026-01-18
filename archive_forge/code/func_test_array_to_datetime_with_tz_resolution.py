from datetime import (
from dateutil.tz.tz import tzoffset
import numpy as np
import pytest
from pandas._libs import (
from pandas._libs.tslibs.dtypes import NpyDatetimeUnit
from pandas._libs.tslibs.np_datetime import OutOfBoundsDatetime
from pandas import Timestamp
import pandas._testing as tm
def test_array_to_datetime_with_tz_resolution(self):
    tz = tzoffset('custom', 3600)
    vals = np.array(['2016-01-01 02:03:04.567', NaT], dtype=object)
    res = tslib.array_to_datetime_with_tz(vals, tz, False, False, creso_infer)
    assert res.dtype == 'M8[ms]'
    vals2 = np.array([datetime(2016, 1, 1, 2, 3, 4), NaT], dtype=object)
    res2 = tslib.array_to_datetime_with_tz(vals2, tz, False, False, creso_infer)
    assert res2.dtype == 'M8[us]'
    vals3 = np.array([NaT, np.datetime64(12345, 's')], dtype=object)
    res3 = tslib.array_to_datetime_with_tz(vals3, tz, False, False, creso_infer)
    assert res3.dtype == 'M8[s]'
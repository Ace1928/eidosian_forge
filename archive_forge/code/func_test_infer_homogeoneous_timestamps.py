from datetime import (
from dateutil.tz.tz import tzoffset
import numpy as np
import pytest
from pandas._libs import (
from pandas._libs.tslibs.dtypes import NpyDatetimeUnit
from pandas._libs.tslibs.np_datetime import OutOfBoundsDatetime
from pandas import Timestamp
import pandas._testing as tm
def test_infer_homogeoneous_timestamps(self):
    dt = datetime(2023, 10, 27, 18, 3, 5, 678000)
    ts = Timestamp(dt).as_unit('ns')
    arr = np.array([None, ts, ts, ts], dtype=object)
    result, tz = tslib.array_to_datetime(arr, creso=creso_infer)
    assert tz is None
    expected = np.array([np.datetime64('NaT')] + [ts.asm8] * 3, dtype='M8[ns]')
    tm.assert_numpy_array_equal(result, expected)
from datetime import (
from dateutil.tz.tz import tzoffset
import numpy as np
import pytest
from pandas._libs import (
from pandas._libs.tslibs.dtypes import NpyDatetimeUnit
from pandas._libs.tslibs.np_datetime import OutOfBoundsDatetime
from pandas import Timestamp
import pandas._testing as tm
def test_infer_heterogeneous(self):
    dtstr = '2023-10-27 18:03:05.678000'
    arr = np.array([dtstr, dtstr[:-3], dtstr[:-7], None], dtype=object)
    result, tz = tslib.array_to_datetime(arr, creso=creso_infer)
    assert tz is None
    expected = np.array(arr, dtype='M8[us]')
    tm.assert_numpy_array_equal(result, expected)
    result, tz = tslib.array_to_datetime(arr[::-1], creso=creso_infer)
    assert tz is None
    tm.assert_numpy_array_equal(result, expected[::-1])
from datetime import (
from dateutil.tz.tz import tzoffset
import numpy as np
import pytest
from pandas._libs import (
from pandas._libs.tslibs.dtypes import NpyDatetimeUnit
from pandas._libs.tslibs.np_datetime import OutOfBoundsDatetime
from pandas import Timestamp
import pandas._testing as tm
def test_array_to_datetime_with_tz_resolution_all_nat(self):
    tz = tzoffset('custom', 3600)
    vals = np.array(['NaT'], dtype=object)
    res = tslib.array_to_datetime_with_tz(vals, tz, False, False, creso_infer)
    assert res.dtype == 'M8[s]'
    vals2 = np.array([NaT, NaT], dtype=object)
    res2 = tslib.array_to_datetime_with_tz(vals2, tz, False, False, creso_infer)
    assert res2.dtype == 'M8[s]'
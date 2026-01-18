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
def test_astype_ns_to_ms_near_bounds(self):
    ts = pd.Timestamp('1677-09-21 00:12:43.145225')
    target = ts.as_unit('ms')
    dta = DatetimeArray._from_sequence([ts], dtype='M8[ns]')
    assert (dta.view('i8') == ts.as_unit('ns').value).all()
    result = dta.astype('M8[ms]')
    assert result[0] == target
    expected = DatetimeArray._from_sequence([ts], dtype='M8[ms]')
    assert (expected.view('i8') == target._value).all()
    tm.assert_datetime_array_equal(result, expected)
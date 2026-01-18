from datetime import (
from importlib import reload
import string
import sys
import numpy as np
import pytest
from pandas._libs.tslibs import iNaT
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('dtype', tm.ALL_INT_EA_DTYPES + tm.FLOAT_EA_DTYPES)
def test_astype_ea_to_datetimetzdtype(self, dtype):
    ser = Series([4, 0, 9], dtype=dtype)
    result = ser.astype(DatetimeTZDtype(tz='US/Pacific'))
    expected = Series({0: Timestamp('1969-12-31 16:00:00.000000004-08:00', tz='US/Pacific'), 1: Timestamp('1969-12-31 16:00:00.000000000-08:00', tz='US/Pacific'), 2: Timestamp('1969-12-31 16:00:00.000000009-08:00', tz='US/Pacific')})
    tm.assert_series_equal(result, expected)
from datetime import datetime
import pytest
import pytz
from pandas.errors import NullFrequencyError
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_dti_shift_across_dst(self, unit):
    idx = date_range('2013-11-03', tz='America/Chicago', periods=7, freq='h', unit=unit)
    ser = Series(index=idx[:-1], dtype=object)
    result = ser.shift(freq='h')
    expected = Series(index=idx[1:], dtype=object)
    tm.assert_series_equal(result, expected)
from datetime import (
from itertools import (
import operator
import numpy as np
import pytest
import pytz
from pandas._libs.tslibs.conversion import localize_pydatetime
from pandas._libs.tslibs.offsets import shift_months
from pandas.errors import PerformanceWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import roperator
from pandas.tests.arithmetic.common import (
def test_dt64arr_sub_timestamp_tzaware(self, box_with_array):
    ser = date_range('2014-03-17', periods=2, freq='D', tz='US/Eastern')
    ser = ser._with_freq(None)
    ts = ser[0]
    ser = tm.box_expected(ser, box_with_array)
    delta_series = Series([np.timedelta64(0, 'D'), np.timedelta64(1, 'D')])
    expected = tm.box_expected(delta_series, box_with_array)
    tm.assert_equal(ser - ts, expected)
    tm.assert_equal(ts - ser, -expected)
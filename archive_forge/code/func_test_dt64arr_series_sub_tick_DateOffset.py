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
def test_dt64arr_series_sub_tick_DateOffset(self, box_with_array):
    ser = Series([Timestamp('20130101 9:01'), Timestamp('20130101 9:02')])
    expected = Series([Timestamp('20130101 9:00:55'), Timestamp('20130101 9:01:55')])
    ser = tm.box_expected(ser, box_with_array)
    expected = tm.box_expected(expected, box_with_array)
    result = ser - pd.offsets.Second(5)
    tm.assert_equal(result, expected)
    result2 = -pd.offsets.Second(5) + ser
    tm.assert_equal(result2, expected)
    msg = '(bad|unsupported) operand type for unary'
    with pytest.raises(TypeError, match=msg):
        pd.offsets.Second(5) - ser
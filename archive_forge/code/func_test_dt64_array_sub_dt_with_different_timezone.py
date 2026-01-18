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
def test_dt64_array_sub_dt_with_different_timezone(self, box_with_array):
    t1 = date_range('20130101', periods=3).tz_localize('US/Eastern')
    t1 = tm.box_expected(t1, box_with_array)
    t2 = Timestamp('20130101').tz_localize('CET')
    tnaive = Timestamp(20130101)
    result = t1 - t2
    expected = TimedeltaIndex(['0 days 06:00:00', '1 days 06:00:00', '2 days 06:00:00'])
    expected = tm.box_expected(expected, box_with_array)
    tm.assert_equal(result, expected)
    result = t2 - t1
    expected = TimedeltaIndex(['-1 days +18:00:00', '-2 days +18:00:00', '-3 days +18:00:00'])
    expected = tm.box_expected(expected, box_with_array)
    tm.assert_equal(result, expected)
    msg = 'Cannot subtract tz-naive and tz-aware datetime-like objects'
    with pytest.raises(TypeError, match=msg):
        t1 - tnaive
    with pytest.raises(TypeError, match=msg):
        tnaive - t1
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
@pytest.mark.parametrize('datetimelike', [Timestamp('20130101'), datetime(2013, 1, 1), np.datetime64('2013-01-01T00:00', 'ns')])
@pytest.mark.parametrize('op,expected', [(operator.lt, [True, False, False, False]), (operator.le, [True, True, False, False]), (operator.eq, [False, True, False, False]), (operator.gt, [False, False, False, True])])
def test_dt64_compare_datetime_scalar(self, datetimelike, op, expected):
    ser = Series([Timestamp('20120101'), Timestamp('20130101'), np.nan, Timestamp('20130103')], name='A')
    result = op(ser, datetimelike)
    expected = Series(expected, name='A')
    tm.assert_series_equal(result, expected)
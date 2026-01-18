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
@pytest.mark.filterwarnings('ignore::pandas.errors.PerformanceWarning')
@pytest.mark.parametrize('cls_and_kwargs', ['YearBegin', ('YearBegin', {'month': 5}), 'YearEnd', ('YearEnd', {'month': 5}), 'MonthBegin', 'MonthEnd', 'SemiMonthEnd', 'SemiMonthBegin', 'Week', ('Week', {'weekday': 3}), 'Week', ('Week', {'weekday': 6}), 'BusinessDay', 'BDay', 'QuarterEnd', 'QuarterBegin', 'CustomBusinessDay', 'CDay', 'CBMonthEnd', 'CBMonthBegin', 'BMonthBegin', 'BMonthEnd', 'BusinessHour', 'BYearBegin', 'BYearEnd', 'BQuarterBegin', ('LastWeekOfMonth', {'weekday': 2}), ('FY5253Quarter', {'qtr_with_extra_week': 1, 'startingMonth': 1, 'weekday': 2, 'variation': 'nearest'}), ('FY5253', {'weekday': 0, 'startingMonth': 2, 'variation': 'nearest'}), ('WeekOfMonth', {'weekday': 2, 'week': 2}), 'Easter', ('DateOffset', {'day': 4}), ('DateOffset', {'month': 5})])
@pytest.mark.parametrize('normalize', [True, False])
@pytest.mark.parametrize('n', [0, 5])
@pytest.mark.parametrize('unit', ['s', 'ms', 'us', 'ns'])
@pytest.mark.parametrize('tz', [None, 'US/Central'])
def test_dt64arr_add_sub_DateOffsets(self, box_with_array, n, normalize, cls_and_kwargs, unit, tz):
    if isinstance(cls_and_kwargs, tuple):
        cls_name, kwargs = cls_and_kwargs
    else:
        cls_name = cls_and_kwargs
        kwargs = {}
    if n == 0 and cls_name in ['WeekOfMonth', 'LastWeekOfMonth', 'FY5253Quarter', 'FY5253']:
        return
    vec = DatetimeIndex([Timestamp('2000-01-05 00:15:00'), Timestamp('2000-01-31 00:23:00'), Timestamp('2000-01-01'), Timestamp('2000-03-31'), Timestamp('2000-02-29'), Timestamp('2000-12-31'), Timestamp('2000-05-15'), Timestamp('2001-06-15')]).as_unit(unit).tz_localize(tz)
    vec = tm.box_expected(vec, box_with_array)
    vec_items = vec.iloc[0] if box_with_array is pd.DataFrame else vec
    offset_cls = getattr(pd.offsets, cls_name)
    offset = offset_cls(n, normalize=normalize, **kwargs)
    expected = DatetimeIndex([x + offset for x in vec_items]).as_unit(unit)
    expected = tm.box_expected(expected, box_with_array)
    tm.assert_equal(expected, vec + offset)
    tm.assert_equal(expected, offset + vec)
    expected = DatetimeIndex([x - offset for x in vec_items]).as_unit(unit)
    expected = tm.box_expected(expected, box_with_array)
    tm.assert_equal(expected, vec - offset)
    expected = DatetimeIndex([offset + x for x in vec_items]).as_unit(unit)
    expected = tm.box_expected(expected, box_with_array)
    tm.assert_equal(expected, offset + vec)
    msg = '(bad|unsupported) operand type for unary'
    with pytest.raises(TypeError, match=msg):
        offset - vec
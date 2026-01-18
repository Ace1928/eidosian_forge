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
def test_dti_add_tick_tzaware(self, tz_aware_fixture, box_with_array):
    tz = tz_aware_fixture
    if tz == 'US/Pacific':
        dates = date_range('2012-11-01', periods=3, tz=tz)
        offset = dates + pd.offsets.Hour(5)
        assert dates[0] + pd.offsets.Hour(5) == offset[0]
    dates = date_range('2010-11-01 00:00', periods=3, tz=tz, freq='h')
    expected = DatetimeIndex(['2010-11-01 05:00', '2010-11-01 06:00', '2010-11-01 07:00'], freq='h', tz=tz).as_unit('ns')
    dates = tm.box_expected(dates, box_with_array)
    expected = tm.box_expected(expected, box_with_array)
    for scalar in [pd.offsets.Hour(5), np.timedelta64(5, 'h'), timedelta(hours=5)]:
        offset = dates + scalar
        tm.assert_equal(offset, expected)
        offset = scalar + dates
        tm.assert_equal(offset, expected)
        roundtrip = offset - scalar
        tm.assert_equal(roundtrip, dates)
        msg = '|'.join(['bad operand type for unary -', 'cannot subtract DatetimeArray'])
        with pytest.raises(TypeError, match=msg):
            scalar - dates
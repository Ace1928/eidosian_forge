from __future__ import annotations
from datetime import (
from functools import partial
from operator import attrgetter
import dateutil
import dateutil.tz
from dateutil.tz import gettz
import numpy as np
import pytest
import pytz
from pandas._libs.tslibs import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import period_array
def test_construction_dti_with_mixed_timezones(self):
    result = DatetimeIndex([Timestamp('2011-01-01'), Timestamp('2011-01-02')], name='idx')
    exp = DatetimeIndex([Timestamp('2011-01-01'), Timestamp('2011-01-02')], name='idx')
    tm.assert_index_equal(result, exp, exact=True)
    assert isinstance(result, DatetimeIndex)
    result = DatetimeIndex([Timestamp('2011-01-01 10:00', tz='Asia/Tokyo'), Timestamp('2011-01-02 10:00', tz='Asia/Tokyo')], name='idx')
    exp = DatetimeIndex([Timestamp('2011-01-01 10:00'), Timestamp('2011-01-02 10:00')], tz='Asia/Tokyo', name='idx')
    tm.assert_index_equal(result, exp, exact=True)
    assert isinstance(result, DatetimeIndex)
    result = DatetimeIndex([Timestamp('2011-01-01 10:00', tz='US/Eastern'), Timestamp('2011-08-01 10:00', tz='US/Eastern')], name='idx')
    exp = DatetimeIndex([Timestamp('2011-01-01 10:00'), Timestamp('2011-08-01 10:00')], tz='US/Eastern', name='idx')
    tm.assert_index_equal(result, exp, exact=True)
    assert isinstance(result, DatetimeIndex)
    msg = 'cannot be converted to datetime64'
    with pytest.raises(ValueError, match=msg):
        DatetimeIndex([Timestamp('2011-01-01 10:00', tz='Asia/Tokyo'), Timestamp('2011-01-02 10:00', tz='US/Eastern')], name='idx')
    dti = DatetimeIndex([Timestamp('2011-01-01 10:00'), Timestamp('2011-01-02 10:00', tz='US/Eastern')], tz='Asia/Tokyo', name='idx')
    expected = DatetimeIndex([Timestamp('2011-01-01 10:00', tz='Asia/Tokyo'), Timestamp('2011-01-02 10:00', tz='US/Eastern').tz_convert('Asia/Tokyo')], tz='Asia/Tokyo', name='idx')
    tm.assert_index_equal(dti, expected)
    dti = DatetimeIndex([Timestamp('2011-01-01 10:00', tz='Asia/Tokyo'), Timestamp('2011-01-02 10:00', tz='US/Eastern')], tz='US/Eastern', name='idx')
    expected = DatetimeIndex([Timestamp('2011-01-01 10:00', tz='Asia/Tokyo').tz_convert('US/Eastern'), Timestamp('2011-01-02 10:00', tz='US/Eastern')], tz='US/Eastern', name='idx')
    tm.assert_index_equal(dti, expected)
    dti = DatetimeIndex([Timestamp('2011-01-01 10:00', tz='Asia/Tokyo'), Timestamp('2011-01-02 10:00', tz='US/Eastern')], dtype='M8[ns, US/Eastern]', name='idx')
    tm.assert_index_equal(dti, expected)
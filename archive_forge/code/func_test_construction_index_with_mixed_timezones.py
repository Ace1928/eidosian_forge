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
def test_construction_index_with_mixed_timezones(self):
    result = Index([Timestamp('2011-01-01'), Timestamp('2011-01-02')], name='idx')
    exp = DatetimeIndex([Timestamp('2011-01-01'), Timestamp('2011-01-02')], name='idx')
    tm.assert_index_equal(result, exp, exact=True)
    assert isinstance(result, DatetimeIndex)
    assert result.tz is None
    result = Index([Timestamp('2011-01-01 10:00', tz='Asia/Tokyo'), Timestamp('2011-01-02 10:00', tz='Asia/Tokyo')], name='idx')
    exp = DatetimeIndex([Timestamp('2011-01-01 10:00'), Timestamp('2011-01-02 10:00')], tz='Asia/Tokyo', name='idx')
    tm.assert_index_equal(result, exp, exact=True)
    assert isinstance(result, DatetimeIndex)
    assert result.tz is not None
    assert result.tz == exp.tz
    result = Index([Timestamp('2011-01-01 10:00', tz='US/Eastern'), Timestamp('2011-08-01 10:00', tz='US/Eastern')], name='idx')
    exp = DatetimeIndex([Timestamp('2011-01-01 10:00'), Timestamp('2011-08-01 10:00')], tz='US/Eastern', name='idx')
    tm.assert_index_equal(result, exp, exact=True)
    assert isinstance(result, DatetimeIndex)
    assert result.tz is not None
    assert result.tz == exp.tz
    result = Index([Timestamp('2011-01-01 10:00'), Timestamp('2011-01-02 10:00', tz='US/Eastern')], name='idx')
    exp = Index([Timestamp('2011-01-01 10:00'), Timestamp('2011-01-02 10:00', tz='US/Eastern')], dtype='object', name='idx')
    tm.assert_index_equal(result, exp, exact=True)
    assert not isinstance(result, DatetimeIndex)
    result = Index([Timestamp('2011-01-01 10:00', tz='Asia/Tokyo'), Timestamp('2011-01-02 10:00', tz='US/Eastern')], name='idx')
    exp = Index([Timestamp('2011-01-01 10:00', tz='Asia/Tokyo'), Timestamp('2011-01-02 10:00', tz='US/Eastern')], dtype='object', name='idx')
    tm.assert_index_equal(result, exp, exact=True)
    assert not isinstance(result, DatetimeIndex)
    msg = 'DatetimeIndex has mixed timezones'
    msg_depr = 'parsing datetimes with mixed time zones will raise an error'
    with pytest.raises(TypeError, match=msg):
        with tm.assert_produces_warning(FutureWarning, match=msg_depr):
            DatetimeIndex(['2013-11-02 22:00-05:00', '2013-11-03 22:00-06:00'])
    result = Index([Timestamp('2011-01-01')], name='idx')
    exp = DatetimeIndex([Timestamp('2011-01-01')], name='idx')
    tm.assert_index_equal(result, exp, exact=True)
    assert isinstance(result, DatetimeIndex)
    assert result.tz is None
    result = Index([Timestamp('2011-01-01 10:00', tz='Asia/Tokyo')], name='idx')
    exp = DatetimeIndex([Timestamp('2011-01-01 10:00')], tz='Asia/Tokyo', name='idx')
    tm.assert_index_equal(result, exp, exact=True)
    assert isinstance(result, DatetimeIndex)
    assert result.tz is not None
    assert result.tz == exp.tz
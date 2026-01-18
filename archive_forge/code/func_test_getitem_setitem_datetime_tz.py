from datetime import (
import re
from dateutil.tz import (
import numpy as np
import pytest
import pytz
from pandas._libs import index as libindex
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('tz_source', ['pytz', 'dateutil'])
def test_getitem_setitem_datetime_tz(tz_source):
    if tz_source == 'pytz':
        tzget = pytz.timezone
    else:
        tzget = lambda x: tzutc() if x == 'UTC' else gettz(x)
    N = 50
    rng = date_range('1/1/1990', periods=N, freq='h', tz=tzget('US/Eastern'))
    ts = Series(np.random.default_rng(2).standard_normal(N), index=rng)
    result = ts.copy()
    result['1990-01-01 09:00:00+00:00'] = 0
    result['1990-01-01 09:00:00+00:00'] = ts.iloc[4]
    tm.assert_series_equal(result, ts)
    result = ts.copy()
    result['1990-01-01 03:00:00-06:00'] = 0
    result['1990-01-01 03:00:00-06:00'] = ts.iloc[4]
    tm.assert_series_equal(result, ts)
    result = ts.copy()
    result[datetime(1990, 1, 1, 9, tzinfo=tzget('UTC'))] = 0
    result[datetime(1990, 1, 1, 9, tzinfo=tzget('UTC'))] = ts.iloc[4]
    tm.assert_series_equal(result, ts)
    result = ts.copy()
    dt = Timestamp(1990, 1, 1, 3).tz_localize(tzget('US/Central'))
    dt = dt.to_pydatetime()
    result[dt] = 0
    result[dt] = ts.iloc[4]
    tm.assert_series_equal(result, ts)
from datetime import (
import numpy as np
import pytest
from pandas._libs.tslibs import timezones
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('tzstr', ['US/Eastern', 'dateutil/US/Eastern'])
def test_localized_between_time(self, tzstr, frame_or_series):
    tz = timezones.maybe_get_tz(tzstr)
    rng = date_range('4/16/2012', '5/1/2012', freq='h')
    ts = Series(np.random.default_rng(2).standard_normal(len(rng)), index=rng)
    if frame_or_series is DataFrame:
        ts = ts.to_frame()
    ts_local = ts.tz_localize(tzstr)
    t1, t2 = (time(10, 0), time(11, 0))
    result = ts_local.between_time(t1, t2)
    expected = ts.between_time(t1, t2).tz_localize(tzstr)
    tm.assert_equal(result, expected)
    assert timezones.tz_compare(result.index.tz, tz)
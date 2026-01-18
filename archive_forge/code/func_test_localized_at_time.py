from datetime import time
import numpy as np
import pytest
import pytz
from pandas._libs.tslibs import timezones
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('tzstr', ['US/Eastern', 'dateutil/US/Eastern'])
def test_localized_at_time(self, tzstr, frame_or_series):
    tz = timezones.maybe_get_tz(tzstr)
    rng = date_range('4/16/2012', '5/1/2012', freq='h')
    ts = frame_or_series(np.random.default_rng(2).standard_normal(len(rng)), index=rng)
    ts_local = ts.tz_localize(tzstr)
    result = ts_local.at_time(time(10, 0))
    expected = ts.at_time(time(10, 0)).tz_localize(tzstr)
    tm.assert_equal(result, expected)
    assert timezones.tz_compare(result.index.tz, tz)
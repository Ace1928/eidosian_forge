from datetime import (
import dateutil.tz
from dateutil.tz import gettz
import numpy as np
import pytest
import pytz
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('start_ts, tz, end_ts, shift', [['2015-03-29 02:20:00', 'Europe/Warsaw', '2015-03-29 03:00:00', 'forward'], ['2015-03-29 02:20:00', 'Europe/Warsaw', '2015-03-29 01:59:59.999999999', 'backward'], ['2015-03-29 02:20:00', 'Europe/Warsaw', '2015-03-29 03:20:00', timedelta(hours=1)], ['2015-03-29 02:20:00', 'Europe/Warsaw', '2015-03-29 01:20:00', timedelta(hours=-1)], ['2018-03-11 02:33:00', 'US/Pacific', '2018-03-11 03:00:00', 'forward'], ['2018-03-11 02:33:00', 'US/Pacific', '2018-03-11 01:59:59.999999999', 'backward'], ['2018-03-11 02:33:00', 'US/Pacific', '2018-03-11 03:33:00', timedelta(hours=1)], ['2018-03-11 02:33:00', 'US/Pacific', '2018-03-11 01:33:00', timedelta(hours=-1)]])
@pytest.mark.parametrize('tz_type', ['', 'dateutil/'])
def test_dti_tz_localize_nonexistent_shift(self, start_ts, tz, end_ts, shift, tz_type, unit):
    tz = tz_type + tz
    if isinstance(shift, str):
        shift = 'shift_' + shift
    dti = DatetimeIndex([Timestamp(start_ts)]).as_unit(unit)
    result = dti.tz_localize(tz, nonexistent=shift)
    expected = DatetimeIndex([Timestamp(end_ts)]).tz_localize(tz).as_unit(unit)
    tm.assert_index_equal(result, expected)
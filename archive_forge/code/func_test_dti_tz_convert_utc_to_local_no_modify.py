from datetime import datetime
import dateutil.tz
from dateutil.tz import gettz
import numpy as np
import pytest
import pytz
from pandas._libs.tslibs import timezones
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('tz', ['US/Eastern', 'dateutil/US/Eastern', pytz.timezone('US/Eastern'), gettz('US/Eastern')])
def test_dti_tz_convert_utc_to_local_no_modify(self, tz):
    rng = date_range('3/11/2012', '3/12/2012', freq='h', tz='utc')
    rng_eastern = rng.tz_convert(tz)
    tm.assert_numpy_array_equal(rng.asi8, rng_eastern.asi8)
    assert timezones.tz_compare(rng_eastern.tz, timezones.maybe_get_tz(tz))
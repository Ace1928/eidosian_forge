from datetime import (
import dateutil.tz
from dateutil.tz import gettz
import numpy as np
import pytest
import pytz
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('tzstr', ['US/Eastern', 'dateutil/US/Eastern'])
def test_dti_tz_localize_pass_dates_to_utc(self, tzstr):
    strdates = ['1/1/2012', '3/1/2012', '4/1/2012']
    idx = DatetimeIndex(strdates)
    conv = idx.tz_localize(tzstr)
    fromdates = DatetimeIndex(strdates, tz=tzstr)
    assert conv.tz == fromdates.tz
    tm.assert_numpy_array_equal(conv.values, fromdates.values)
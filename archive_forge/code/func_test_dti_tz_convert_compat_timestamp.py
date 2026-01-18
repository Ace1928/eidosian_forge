from datetime import datetime
import dateutil.tz
from dateutil.tz import gettz
import numpy as np
import pytest
import pytz
from pandas._libs.tslibs import timezones
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('prefix', ['', 'dateutil/'])
def test_dti_tz_convert_compat_timestamp(self, prefix):
    strdates = ['1/1/2012', '3/1/2012', '4/1/2012']
    idx = DatetimeIndex(strdates, tz=prefix + 'US/Eastern')
    conv = idx[0].tz_convert(prefix + 'US/Pacific')
    expected = idx.tz_convert(prefix + 'US/Pacific')[0]
    assert conv == expected
from datetime import datetime
import dateutil.tz
from dateutil.tz import gettz
import numpy as np
import pytest
import pytz
from pandas._libs.tslibs import timezones
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('freq, n', [('h', 1), ('min', 60), ('s', 3600)])
def test_dti_tz_convert_trans_pos_plus_1__bug(self, freq, n):
    idx = date_range(datetime(2011, 3, 26, 23), datetime(2011, 3, 27, 1), freq=freq)
    idx = idx.tz_localize('UTC')
    idx = idx.tz_convert('Europe/Moscow')
    expected = np.repeat(np.array([3, 4, 5]), np.array([n, n, 1]))
    tm.assert_index_equal(idx.hour, Index(expected, dtype=np.int32))
from datetime import datetime
import warnings
import dateutil
import numpy as np
import pytest
import pytz
from pandas._libs.tslibs.ccalendar import (
from pandas._libs.tslibs.period import IncompatibleFrequency
from pandas.errors import InvalidIndexError
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.indexes.datetimes import date_range
from pandas.core.indexes.period import (
from pandas.core.resample import _get_period_range_edges
from pandas.tseries import offsets
@pytest.mark.parametrize('freq', ['D', '2D'])
def test_basic_upsample(self, freq, simple_period_range_series):
    ts = simple_period_range_series('1/1/1990', '6/30/1995', freq='M')
    result = ts.resample('Y-DEC').mean()
    msg = "The 'convention' keyword in Series.resample is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        resampled = result.resample(freq, convention='end').ffill()
    expected = result.to_timestamp(freq, how='end')
    expected = expected.asfreq(freq, 'ffill').to_period(freq)
    tm.assert_series_equal(resampled, expected)
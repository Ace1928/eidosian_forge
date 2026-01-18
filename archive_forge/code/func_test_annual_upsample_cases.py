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
@pytest.mark.parametrize('month', MONTHS)
@pytest.mark.parametrize('meth', ['ffill', 'bfill'])
@pytest.mark.parametrize('conv', ['start', 'end'])
@pytest.mark.parametrize(('offset', 'period'), [('D', 'D'), ('B', 'B'), ('ME', 'M'), ('QE', 'Q')])
def test_annual_upsample_cases(self, offset, period, conv, meth, month, simple_period_range_series):
    ts = simple_period_range_series('1/1/1990', '12/31/1991', freq=f'Y-{month}')
    warn = FutureWarning if period == 'B' else None
    msg = 'PeriodDtype\\[B\\] is deprecated'
    if warn is None:
        msg = 'Resampling with a PeriodIndex is deprecated'
        warn = FutureWarning
    with tm.assert_produces_warning(warn, match=msg):
        result = getattr(ts.resample(period, convention=conv), meth)()
        expected = result.to_timestamp(period, how=conv)
        expected = expected.asfreq(offset, meth).to_period()
    tm.assert_series_equal(result, expected)
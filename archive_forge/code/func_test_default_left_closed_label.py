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
@pytest.mark.parametrize('from_freq, to_freq', [('D', 'MS'), ('QE', 'YS'), ('ME', 'QS'), ('h', 'D'), ('min', 'h')])
def test_default_left_closed_label(self, from_freq, to_freq):
    idx = date_range(start='8/15/2012', periods=100, freq=from_freq)
    df = DataFrame(np.random.default_rng(2).standard_normal((len(idx), 2)), idx)
    resampled = df.resample(to_freq).mean()
    tm.assert_frame_equal(resampled, df.resample(to_freq, closed='left', label='left').mean())
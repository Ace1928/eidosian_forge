from datetime import (
import pickle
import numpy as np
import pytest
from pandas._libs.tslibs import (
from pandas._libs.tslibs.dtypes import freq_to_period_freqstr
from pandas import (
import pandas._testing as tm
from pandas.core.indexes.datetimes import (
from pandas.core.indexes.period import (
from pandas.core.indexes.timedeltas import timedelta_range
from pandas.tests.plotting.common import _check_ticks_props
from pandas.tseries.offsets import WeekOfMonth
def test_mixed_freq_alignment(self):
    ts_ind = date_range('2012-01-01 13:00', '2012-01-02', freq='h')
    ts_data = np.random.default_rng(2).standard_normal(12)
    ts = Series(ts_data, index=ts_ind)
    ts2 = ts.asfreq('min').interpolate()
    _, ax = mpl.pyplot.subplots()
    ax = ts.plot(ax=ax)
    ts2.plot(style='r', ax=ax)
    assert ax.lines[0].get_xdata()[0] == ax.lines[1].get_xdata()[0]
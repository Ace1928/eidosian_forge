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
@pytest.mark.parametrize('freq', ['ms', 'us'])
def test_high_freq(self, freq):
    _, ax = mpl.pyplot.subplots()
    rng = date_range('1/1/2012', periods=100, freq=freq)
    ser = Series(np.random.default_rng(2).standard_normal(len(rng)), rng)
    _check_plot_works(ser.plot, ax=ax)
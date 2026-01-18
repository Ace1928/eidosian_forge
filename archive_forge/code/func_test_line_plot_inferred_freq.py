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
@pytest.mark.parametrize('freq', ['s', 'min', 'h', 'D', 'W', 'ME', 'QE-DEC', 'YE', '1B30Min'])
def test_line_plot_inferred_freq(self, freq):
    idx = date_range('12/31/1999', freq=freq, periods=100)
    ser = Series(np.random.default_rng(2).standard_normal(len(idx)), idx)
    ser = Series(ser.values, Index(np.asarray(ser.index)))
    _check_plot_works(ser.plot, ser.index.inferred_freq)
    ser = ser.iloc[[0, 3, 5, 6]]
    _check_plot_works(ser.plot)
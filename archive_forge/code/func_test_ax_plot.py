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
def test_ax_plot(self):
    x = date_range(start='2012-01-02', periods=10, freq='D')
    y = list(range(len(x)))
    _, ax = mpl.pyplot.subplots()
    lines = ax.plot(x, y, label='Y')
    tm.assert_index_equal(DatetimeIndex(lines[0].get_xdata()), x)
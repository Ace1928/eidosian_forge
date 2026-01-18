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
def test_matplotlib_scatter_datetime64(self):
    df = DataFrame(np.random.default_rng(2).random((10, 2)), columns=['x', 'y'])
    df['time'] = date_range('2018-01-01', periods=10, freq='D')
    _, ax = mpl.pyplot.subplots()
    ax.scatter(x='time', y='y', data=df)
    mpl.pyplot.draw()
    label = ax.get_xticklabels()[0]
    expected = '2018-01-01'
    assert label.get_text() == expected
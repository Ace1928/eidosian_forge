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
def test_business_freq_convert(self):
    bts = Series(np.arange(300, dtype=np.float64), index=date_range('2020-01-01', periods=300, freq='B')).asfreq('BME')
    ts = bts.to_period('M')
    _, ax = mpl.pyplot.subplots()
    bts.plot(ax=ax)
    assert ax.get_lines()[0].get_xydata()[0, 0] == ts.index[0].ordinal
    idx = ax.get_lines()[0].get_xdata()
    assert PeriodIndex(data=idx).freqstr == 'M'
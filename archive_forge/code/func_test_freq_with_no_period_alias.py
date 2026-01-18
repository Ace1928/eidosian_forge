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
def test_freq_with_no_period_alias(self):
    freq = WeekOfMonth()
    bts = Series(np.arange(10, dtype=np.float64), index=date_range('2020-01-01', periods=10)).asfreq(freq)
    _, ax = mpl.pyplot.subplots()
    bts.plot(ax=ax)
    idx = ax.get_lines()[0].get_xdata()
    msg = 'freq not specified and cannot be inferred'
    with pytest.raises(ValueError, match=msg):
        PeriodIndex(data=idx)
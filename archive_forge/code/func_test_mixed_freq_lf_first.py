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
def test_mixed_freq_lf_first(self):
    idxh = date_range('1/1/1999', periods=365, freq='D')
    idxl = date_range('1/1/1999', periods=12, freq='ME')
    high = Series(np.random.default_rng(2).standard_normal(len(idxh)), idxh)
    low = Series(np.random.default_rng(2).standard_normal(len(idxl)), idxl)
    _, ax = mpl.pyplot.subplots()
    low.plot(legend=True, ax=ax)
    high.plot(legend=True, ax=ax)
    for line in ax.get_lines():
        assert PeriodIndex(data=line.get_xdata()).freq == 'D'
    leg = ax.get_legend()
    assert len(leg.texts) == 2
    mpl.pyplot.close(ax.get_figure())
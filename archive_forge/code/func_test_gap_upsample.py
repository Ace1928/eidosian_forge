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
def test_gap_upsample(self):
    low = Series(np.arange(30, dtype=np.float64), index=date_range('2020-01-01', periods=30))
    low.iloc[5:25] = np.nan
    _, ax = mpl.pyplot.subplots()
    low.plot(ax=ax)
    idxh = date_range(low.index[0], low.index[-1], freq='12h')
    s = Series(np.random.default_rng(2).standard_normal(len(idxh)), idxh)
    s.plot(secondary_y=True)
    lines = ax.get_lines()
    assert len(lines) == 1
    assert len(ax.right_ax.get_lines()) == 1
    line = lines[0]
    data = line.get_xydata()
    data = np.ma.MaskedArray(data, mask=isna(data), fill_value=np.nan)
    assert isinstance(data, np.ma.core.MaskedArray)
    mask = data.mask
    assert mask[5:25, 1].all()
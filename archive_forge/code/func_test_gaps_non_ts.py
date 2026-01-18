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
def test_gaps_non_ts(self):
    idx = [0, 1, 2, 5, 7, 9, 12, 15, 20]
    ser = Series(np.random.default_rng(2).standard_normal(len(idx)), idx)
    ser.iloc[2:5] = np.nan
    _, ax = mpl.pyplot.subplots()
    ser.plot(ax=ax)
    lines = ax.get_lines()
    assert len(lines) == 1
    line = lines[0]
    data = line.get_xydata()
    data = np.ma.MaskedArray(data, mask=isna(data), fill_value=np.nan)
    assert isinstance(data, np.ma.core.MaskedArray)
    mask = data.mask
    assert mask[2:5, 1].all()
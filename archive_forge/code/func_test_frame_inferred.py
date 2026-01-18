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
def test_frame_inferred(self):
    idx = date_range('1/1/1987', freq='MS', periods=100)
    idx = DatetimeIndex(idx.values, freq=None)
    df = DataFrame(np.random.default_rng(2).standard_normal((len(idx), 3)), index=idx)
    _check_plot_works(df.plot)
    idx = idx[0:40].union(idx[45:99])
    df2 = DataFrame(np.random.default_rng(2).standard_normal((len(idx), 3)), index=idx)
    _check_plot_works(df2.plot)
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
def test_frame_inferred_n_gt_1(self):
    idx = date_range('2008-1-1 00:15:00', freq='15min', periods=10)
    idx = DatetimeIndex(idx.values, freq=None)
    df = DataFrame(np.random.default_rng(2).standard_normal((len(idx), 3)), index=idx)
    _check_plot_works(df.plot)
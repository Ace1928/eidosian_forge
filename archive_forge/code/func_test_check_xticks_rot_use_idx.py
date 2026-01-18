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
def test_check_xticks_rot_use_idx(self):
    x = to_datetime(['2020-05-01', '2020-05-02', '2020-05-04'])
    df = DataFrame({'x': x, 'y': [1, 2, 3]})
    axes = df.set_index('x').plot(y='y', use_index=True)
    _check_ticks_props(axes, xrot=30)
    axes = df.set_index('x').plot(y='y', use_index=False)
    _check_ticks_props(axes, xrot=0)
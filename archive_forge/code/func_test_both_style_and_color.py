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
def test_both_style_and_color(self):
    ts = Series(np.arange(10, dtype=np.float64), index=date_range('2020-01-01', periods=10))
    msg = "Cannot pass 'style' string with a color symbol and 'color' keyword argument. Please use one or the other or pass 'style' without a color symbol"
    with pytest.raises(ValueError, match=msg):
        ts.plot(style='b-', color='#000099')
    s = ts.reset_index(drop=True)
    with pytest.raises(ValueError, match=msg):
        s.plot(style='b-', color='#000099')
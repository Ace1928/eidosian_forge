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
@pytest.mark.xfail(reason='GH9053 matplotlib does not use ax.xaxis.converter')
def test_add_matplotlib_datetime64(self):
    s = Series(np.random.default_rng(2).standard_normal(10), index=date_range('1970-01-02', periods=10))
    ax = s.plot()
    with tm.assert_produces_warning(DeprecationWarning):
        ax.plot(s.index, s.values, color='g')
    l1, l2 = ax.lines
    tm.assert_numpy_array_equal(l1.get_xydata(), l2.get_xydata())
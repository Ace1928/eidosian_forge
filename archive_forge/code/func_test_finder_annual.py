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
def test_finder_annual(self):
    xp = [1987, 1988, 1990, 1990, 1995, 2020, 2070, 2170]
    xp = [Period(x, freq='Y').ordinal for x in xp]
    rs = []
    for nyears in [5, 10, 19, 49, 99, 199, 599, 1001]:
        rng = period_range('1987', periods=nyears, freq='Y')
        ser = Series(np.random.default_rng(2).standard_normal(len(rng)), rng)
        _, ax = mpl.pyplot.subplots()
        ser.plot(ax=ax)
        xaxis = ax.get_xaxis()
        rs.append(xaxis.get_majorticklocs()[0])
        mpl.pyplot.close(ax.get_figure())
    assert rs == xp
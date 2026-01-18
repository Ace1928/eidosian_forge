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
def test_finder_daily(self):
    day_lst = [10, 40, 252, 400, 950, 2750, 10000]
    msg = 'Period with BDay freq is deprecated'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        xpl1 = xpl2 = [Period('1999-1-1', freq='B').ordinal] * len(day_lst)
    rs1 = []
    rs2 = []
    for n in day_lst:
        rng = bdate_range('1999-1-1', periods=n)
        ser = Series(np.random.default_rng(2).standard_normal(len(rng)), rng)
        _, ax = mpl.pyplot.subplots()
        ser.plot(ax=ax)
        xaxis = ax.get_xaxis()
        rs1.append(xaxis.get_majorticklocs()[0])
        vmin, vmax = ax.get_xlim()
        ax.set_xlim(vmin + 0.9, vmax)
        rs2.append(xaxis.get_majorticklocs()[0])
        mpl.pyplot.close(ax.get_figure())
    assert rs1 == xpl1
    assert rs2 == xpl2
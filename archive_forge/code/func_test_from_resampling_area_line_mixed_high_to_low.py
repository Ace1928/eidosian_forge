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
@pytest.mark.parametrize('kind1, kind2', [('line', 'area'), ('area', 'line')])
def test_from_resampling_area_line_mixed_high_to_low(self, kind1, kind2):
    idxh = date_range('1/1/1999', periods=52, freq='W')
    idxl = date_range('1/1/1999', periods=12, freq='ME')
    high = DataFrame(np.random.default_rng(2).random((len(idxh), 3)), index=idxh, columns=[0, 1, 2])
    low = DataFrame(np.random.default_rng(2).random((len(idxl), 3)), index=idxl, columns=[0, 1, 2])
    _, ax = mpl.pyplot.subplots()
    high.plot(kind=kind1, stacked=True, ax=ax)
    low.plot(kind=kind2, stacked=True, ax=ax)
    expected_x = idxh.to_period().asi8.astype(np.float64)
    expected_y = np.zeros(len(expected_x), dtype=np.float64)
    for i in range(3):
        line = ax.lines[i]
        assert PeriodIndex(data=line.get_xdata()).freq == idxh.freq
        tm.assert_numpy_array_equal(line.get_xdata(orig=False), expected_x)
        expected_y += high[i].values
        tm.assert_numpy_array_equal(line.get_ydata(orig=False), expected_y)
    expected_x = np.array([1514, 1519, 1523, 1527, 1531, 1536, 1540, 1544, 1549, 1553, 1558, 1562], dtype=np.float64)
    expected_y = np.zeros(len(expected_x), dtype=np.float64)
    for i in range(3):
        lines = ax.lines[3 + i]
        assert PeriodIndex(data=lines.get_xdata()).freq == idxh.freq
        tm.assert_numpy_array_equal(lines.get_xdata(orig=False), expected_x)
        expected_y += low[i].values
        tm.assert_numpy_array_equal(lines.get_ydata(orig=False), expected_y)
from datetime import datetime
from itertools import chain
import numpy as np
import pytest
from pandas.compat import is_platform_linux
from pandas.compat.numpy import np_version_gte1p24
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.plotting.common import (
@pytest.mark.parametrize('index', [None, date_range('2020-01-01', periods=4)])
def test_line_area_nan_series(self, index):
    values = [1, 2, np.nan, 3]
    d = Series(values, index=index)
    ax = _check_plot_works(d.plot)
    masked = ax.lines[0].get_ydata()
    exp = np.array([1, 2, 3], dtype=np.float64)
    tm.assert_numpy_array_equal(np.delete(masked.data, 2), exp)
    tm.assert_numpy_array_equal(masked.mask, np.array([False, False, True, False]))
    expected = np.array([1, 2, 0, 3], dtype=np.float64)
    ax = _check_plot_works(d.plot, stacked=True)
    tm.assert_numpy_array_equal(ax.lines[0].get_ydata(), expected)
    ax = _check_plot_works(d.plot.area)
    tm.assert_numpy_array_equal(ax.lines[0].get_ydata(), expected)
    ax = _check_plot_works(d.plot.area, stacked=False)
    tm.assert_numpy_array_equal(ax.lines[0].get_ydata(), expected)
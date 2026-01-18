from datetime import (
import gc
import itertools
import re
import string
import weakref
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas.core.dtypes.api import is_list_like
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.plotting.common import (
from pandas.io.formats.printing import pprint_thing
@pytest.mark.parametrize('idx', [range(4), date_range('2023-01-1', freq='D', periods=4)])
@pytest.mark.parametrize('kwargs', [{}, {'stacked': False}])
def test_line_area_nan_df_stacked_area(self, idx, kwargs):
    values1 = [1, 2, np.nan, 3]
    values2 = [3, np.nan, 2, 1]
    df = DataFrame({'a': values1, 'b': values2}, index=idx)
    expected1 = np.array([1, 2, 0, 3], dtype=np.float64)
    expected2 = np.array([3, 0, 2, 1], dtype=np.float64)
    ax = _check_plot_works(df.plot.area, **kwargs)
    tm.assert_numpy_array_equal(ax.lines[0].get_ydata(), expected1)
    if kwargs:
        tm.assert_numpy_array_equal(ax.lines[1].get_ydata(), expected2)
    else:
        tm.assert_numpy_array_equal(ax.lines[1].get_ydata(), expected1 + expected2)
    ax = _check_plot_works(df.plot.area, stacked=False)
    tm.assert_numpy_array_equal(ax.lines[0].get_ydata(), expected1)
    tm.assert_numpy_array_equal(ax.lines[1].get_ydata(), expected2)
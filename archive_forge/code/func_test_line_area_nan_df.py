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
def test_line_area_nan_df(self, idx):
    values1 = [1, 2, np.nan, 3]
    values2 = [3, np.nan, 2, 1]
    df = DataFrame({'a': values1, 'b': values2}, index=idx)
    ax = _check_plot_works(df.plot)
    masked1 = ax.lines[0].get_ydata()
    masked2 = ax.lines[1].get_ydata()
    exp = np.array([1, 2, 3], dtype=np.float64)
    tm.assert_numpy_array_equal(np.delete(masked1.data, 2), exp)
    exp = np.array([3, 2, 1], dtype=np.float64)
    tm.assert_numpy_array_equal(np.delete(masked2.data, 1), exp)
    tm.assert_numpy_array_equal(masked1.mask, np.array([False, False, True, False]))
    tm.assert_numpy_array_equal(masked2.mask, np.array([False, True, False, False]))
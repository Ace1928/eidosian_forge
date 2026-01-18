from datetime import (
from itertools import product
import re
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.errors import PerformanceWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.types import CategoricalDtype
from pandas.core.reshape import reshape as reshape_lib
from pandas.core.reshape.pivot import pivot_table
def test_monthly(self):
    rng = date_range('1/1/2000', '12/31/2004', freq='ME')
    ts = Series(np.arange(len(rng)), index=rng)
    result = pivot_table(DataFrame(ts), index=ts.index.year, columns=ts.index.month)
    result.columns = result.columns.droplevel(0)
    month = np.asarray(ts.index.month)
    expected = {}
    for y in ts.index.year.unique().values:
        mask = ts.index.year == y
        expected[y] = Series(ts.values[mask], index=month[mask])
    expected = DataFrame(expected, dtype=float).T
    tm.assert_frame_equal(result, expected)
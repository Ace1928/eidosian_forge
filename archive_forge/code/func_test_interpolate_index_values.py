import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_interpolate_index_values(self):
    s = Series(np.nan, index=np.sort(np.random.default_rng(2).random(30)))
    s.loc[::3] = np.random.default_rng(2).standard_normal(10)
    vals = s.index.values.astype(float)
    result = s.interpolate(method='index')
    expected = s.copy()
    bad = isna(expected.values)
    good = ~bad
    expected = Series(np.interp(vals[bad], vals[good], s.values[good]), index=s.index[bad])
    tm.assert_series_equal(result[bad], expected)
    other_result = s.interpolate(method='values')
    tm.assert_series_equal(other_result, result)
    tm.assert_series_equal(other_result[bad], expected)
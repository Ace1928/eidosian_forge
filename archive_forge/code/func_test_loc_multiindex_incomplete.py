import numpy as np
import pytest
from pandas.errors import (
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_loc_multiindex_incomplete(self):
    s = Series(np.arange(15, dtype='int64'), MultiIndex.from_product([range(5), ['a', 'b', 'c']]))
    expected = s.loc[:, 'a':'c']
    result = s.loc[0:4, 'a':'c']
    tm.assert_series_equal(result, expected)
    result = s.loc[:4, 'a':'c']
    tm.assert_series_equal(result, expected)
    result = s.loc[0:, 'a':'c']
    tm.assert_series_equal(result, expected)
    s = Series(np.arange(15, dtype='int64'), MultiIndex.from_product([range(5), ['a', 'b', 'c']]))
    expected = s.iloc[[6, 7, 8, 12, 13, 14]]
    result = s.loc[2:4:2, 'a':'c']
    tm.assert_series_equal(result, expected)
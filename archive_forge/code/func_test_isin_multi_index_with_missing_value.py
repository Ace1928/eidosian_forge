import numpy as np
import pytest
from pandas import MultiIndex
import pandas._testing as tm
@pytest.mark.parametrize('labels,expected,level', [([('b', np.nan)], np.array([False, False, True]), None), ([np.nan, 'a'], np.array([True, True, False]), 0), (['d', np.nan], np.array([False, True, True]), 1)])
def test_isin_multi_index_with_missing_value(labels, expected, level):
    midx = MultiIndex.from_arrays([[np.nan, 'a', 'b'], ['c', 'd', np.nan]])
    result = midx.isin(labels, level=level)
    tm.assert_numpy_array_equal(result, expected)
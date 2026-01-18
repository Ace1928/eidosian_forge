import numpy as np
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_get_level_values_na():
    arrays = [[np.nan, np.nan, np.nan], ['a', np.nan, 1]]
    index = MultiIndex.from_arrays(arrays)
    result = index.get_level_values(0)
    expected = Index([np.nan, np.nan, np.nan])
    tm.assert_index_equal(result, expected)
    result = index.get_level_values(1)
    expected = Index(['a', np.nan, 1])
    tm.assert_index_equal(result, expected)
    arrays = [['a', 'b', 'b'], pd.DatetimeIndex([0, 1, pd.NaT])]
    index = MultiIndex.from_arrays(arrays)
    result = index.get_level_values(1)
    expected = pd.DatetimeIndex([0, 1, pd.NaT])
    tm.assert_index_equal(result, expected)
    arrays = [[], []]
    index = MultiIndex.from_arrays(arrays)
    result = index.get_level_values(0)
    expected = Index([], dtype=object)
    tm.assert_index_equal(result, expected)
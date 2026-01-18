import numpy as np
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_get_level_values_int_with_na():
    arrays = [['a', 'b', 'b'], [1, np.nan, 2]]
    index = MultiIndex.from_arrays(arrays)
    result = index.get_level_values(1)
    expected = Index([1, np.nan, 2])
    tm.assert_index_equal(result, expected)
    arrays = [['a', 'b', 'b'], [np.nan, np.nan, 2]]
    index = MultiIndex.from_arrays(arrays)
    result = index.get_level_values(1)
    expected = Index([np.nan, np.nan, 2])
    tm.assert_index_equal(result, expected)
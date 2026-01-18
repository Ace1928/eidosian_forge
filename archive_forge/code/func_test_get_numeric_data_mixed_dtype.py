import numpy as np
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import IntervalArray
def test_get_numeric_data_mixed_dtype(self):
    df = DataFrame({'a': [1, 2, 3], 'b': [True, False, True], 'c': ['foo', 'bar', 'baz'], 'd': [None, None, None], 'e': [3.14, 0.577, 2.773]})
    result = df._get_numeric_data()
    tm.assert_index_equal(result.columns, Index(['a', 'b', 'e']))
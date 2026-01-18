from string import ascii_letters
import numpy as np
import pytest
from pandas.errors import (
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
@td.skip_array_manager_not_yet_implemented
def test_chained_getitem_with_lists(self):
    df = DataFrame({'A': 5 * [np.zeros(3)], 'B': 5 * [np.ones(3)]})
    expected = df['A'].iloc[2]
    result = df.loc[2, 'A']
    tm.assert_numpy_array_equal(result, expected)
    result2 = df.iloc[2]['A']
    tm.assert_numpy_array_equal(result2, expected)
    result3 = df['A'].loc[2]
    tm.assert_numpy_array_equal(result3, expected)
    result4 = df['A'].iloc[2]
    tm.assert_numpy_array_equal(result4, expected)
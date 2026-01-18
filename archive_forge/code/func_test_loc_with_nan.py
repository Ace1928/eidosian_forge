import numpy as np
import pytest
from pandas.errors import (
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_loc_with_nan():
    df = DataFrame({'col': [1, 2, 5], 'ind1': ['a', 'd', np.nan], 'ind2': [1, 4, 5]}).set_index(['ind1', 'ind2'])
    result = df.loc[['a']]
    expected = DataFrame({'col': [1]}, index=MultiIndex.from_tuples([('a', 1)], names=['ind1', 'ind2']))
    tm.assert_frame_equal(result, expected)
    result = df.loc['a']
    expected = DataFrame({'col': [1]}, index=Index([1], name='ind2'))
    tm.assert_frame_equal(result, expected)
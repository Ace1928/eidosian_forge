import numpy as np
import pytest
import pandas as pd
from pandas import MultiIndex
import pandas._testing as tm
def test_boxable_categorical_values():
    cat = pd.Categorical(pd.date_range('2012-01-01', periods=3, freq='h'))
    result = MultiIndex.from_product([['a', 'b', 'c'], cat]).values
    expected = pd.Series([('a', pd.Timestamp('2012-01-01 00:00:00')), ('a', pd.Timestamp('2012-01-01 01:00:00')), ('a', pd.Timestamp('2012-01-01 02:00:00')), ('b', pd.Timestamp('2012-01-01 00:00:00')), ('b', pd.Timestamp('2012-01-01 01:00:00')), ('b', pd.Timestamp('2012-01-01 02:00:00')), ('c', pd.Timestamp('2012-01-01 00:00:00')), ('c', pd.Timestamp('2012-01-01 01:00:00')), ('c', pd.Timestamp('2012-01-01 02:00:00'))]).values
    tm.assert_numpy_array_equal(result, expected)
    result = pd.DataFrame({'a': ['a', 'b', 'c'], 'b': cat, 'c': np.array(cat)}).values
    expected = pd.DataFrame({'a': ['a', 'b', 'c'], 'b': [pd.Timestamp('2012-01-01 00:00:00'), pd.Timestamp('2012-01-01 01:00:00'), pd.Timestamp('2012-01-01 02:00:00')], 'c': [pd.Timestamp('2012-01-01 00:00:00'), pd.Timestamp('2012-01-01 01:00:00'), pd.Timestamp('2012-01-01 02:00:00')]}).values
    tm.assert_numpy_array_equal(result, expected)
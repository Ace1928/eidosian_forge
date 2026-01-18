import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.api.indexers import BaseIndexer
from pandas.core.groupby.groupby import get_groupby
def test_dont_mutate_obj_after_slicing(self):
    df = DataFrame({'id': ['a', 'a', 'b', 'b', 'b'], 'timestamp': date_range('2021-9-1', periods=5, freq='h'), 'y': range(5)})
    grp = df.groupby('id').rolling('1h', on='timestamp')
    result = grp.count()
    expected_df = DataFrame({'timestamp': date_range('2021-9-1', periods=5, freq='h'), 'y': [1.0] * 5}, index=MultiIndex.from_arrays([['a', 'a', 'b', 'b', 'b'], list(range(5))], names=['id', None]))
    tm.assert_frame_equal(result, expected_df)
    result = grp['y'].count()
    expected_series = Series([1.0] * 5, index=MultiIndex.from_arrays([['a', 'a', 'b', 'b', 'b'], date_range('2021-9-1', periods=5, freq='h')], names=['id', 'timestamp']), name='y')
    tm.assert_series_equal(result, expected_series)
    result = grp.count()
    tm.assert_frame_equal(result, expected_df)
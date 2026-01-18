import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.reshape.concat import concat
from pandas.core.reshape.merge import merge
@pytest.mark.parametrize('merge_type', ['left', 'right'])
def test_merge_datetime_multi_index_empty_df(self, merge_type):
    left = DataFrame(data={'data': [1.5, 1.5]}, index=MultiIndex.from_tuples([[Timestamp('1950-01-01'), 'A'], [Timestamp('1950-01-02'), 'B']], names=['date', 'panel']))
    right = DataFrame(index=MultiIndex.from_tuples([], names=['date', 'panel']), columns=['state'])
    expected_index = MultiIndex.from_tuples([[Timestamp('1950-01-01'), 'A'], [Timestamp('1950-01-02'), 'B']], names=['date', 'panel'])
    if merge_type == 'left':
        expected = DataFrame(data={'data': [1.5, 1.5], 'state': np.array([np.nan, np.nan], dtype=object)}, index=expected_index)
        results_merge = left.merge(right, how='left', on=['date', 'panel'])
        results_join = left.join(right, how='left')
    else:
        expected = DataFrame(data={'state': np.array([np.nan, np.nan], dtype=object), 'data': [1.5, 1.5]}, index=expected_index)
        results_merge = right.merge(left, how='right', on=['date', 'panel'])
        results_join = right.join(left, how='right')
    tm.assert_frame_equal(results_merge, expected)
    tm.assert_frame_equal(results_join, expected)
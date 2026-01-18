from itertools import product
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.util.version import Version
@pytest.mark.parametrize('test, columns, expected_names', [('repeat', list('abbde'), ['a', None, 'd', 'b', 'b', 'e']), ('level', list('abcd') + ['level_1'], ['a', None, 'd', 'b', 'c', 'level_1'])])
@pytest.mark.parametrize('as_index', [False, True])
def test_column_label_duplicates(test, columns, expected_names, as_index):
    df = DataFrame([[1, 3, 5, 7, 9], [2, 4, 6, 8, 10]], columns=columns)
    expected_data = [(1, 0, 7, 3, 5, 9), (2, 1, 8, 4, 6, 10)]
    keys = ['a', np.array([0, 1], dtype=np.int64), 'd']
    result = df.groupby(keys, as_index=as_index).value_counts()
    if as_index:
        expected = Series(data=(1, 1), index=MultiIndex.from_tuples(expected_data, names=expected_names), name='count')
        tm.assert_series_equal(result, expected)
    else:
        expected_data = [list(row) + [1] for row in expected_data]
        expected_columns = list(expected_names)
        expected_columns[1] = 'level_1'
        expected_columns.append('count')
        expected = DataFrame(expected_data, columns=expected_columns)
        tm.assert_frame_equal(result, expected)
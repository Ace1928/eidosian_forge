from itertools import product
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.util.version import Version
@pytest.mark.parametrize('dropna, expected_data, expected_index', [(True, [1, 1], MultiIndex.from_arrays([(1, 1), ('Beth', 'John'), ('Louise', 'Smith')], names=['key', 'first_name', 'middle_name'])), (False, [1, 1, 1, 1], MultiIndex(levels=[Index([1]), Index(['Anne', 'Beth', 'John']), Index(['Louise', 'Smith', np.nan])], codes=[[0, 0, 0, 0], [0, 1, 2, 2], [2, 0, 1, 2]], names=['key', 'first_name', 'middle_name']))])
@pytest.mark.parametrize('normalize, name', [(False, 'count'), (True, 'proportion')])
def test_data_frame_value_counts_dropna(names_with_nulls_df, dropna, normalize, name, expected_data, expected_index):
    result_frame = names_with_nulls_df.value_counts(dropna=dropna, normalize=normalize)
    expected = Series(data=expected_data, index=expected_index, name=name)
    if normalize:
        expected /= float(len(expected_data))
    tm.assert_series_equal(result_frame, expected)
    result_frame_groupby = names_with_nulls_df.groupby('key').value_counts(dropna=dropna, normalize=normalize)
    tm.assert_series_equal(result_frame_groupby, expected)
import re
import numpy as np
import pytest
from pandas.errors import InvalidIndexError
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('query, expected', [([-0.5], ([-1], [0])), ([0], ([0], [])), ([0.5], ([0], [])), ([1], ([0, 1], [])), ([1.5], ([0, 1], [])), ([2], ([0, 1, 2], [])), ([2.5], ([1, 2], [])), ([3], ([2], [])), ([3.5], ([2], [])), ([4], ([-1], [0])), ([4.5], ([-1], [0])), ([1, 2], ([0, 1, 0, 1, 2], [])), ([1, 2, 3], ([0, 1, 0, 1, 2, 2], [])), ([1, 2, 3, 4], ([0, 1, 0, 1, 2, 2, -1], [3])), ([1, 2, 3, 4, 2], ([0, 1, 0, 1, 2, 2, -1, 0, 1, 2], [3]))])
def test_get_indexer_non_unique_with_int_and_float(self, query, expected):
    tuples = [(0, 2.5), (1, 3), (2, 4)]
    index = IntervalIndex.from_tuples(tuples, closed='left')
    result_indexer, result_missing = index.get_indexer_non_unique(query)
    expected_indexer = np.array(expected[0], dtype='intp')
    expected_missing = np.array(expected[1], dtype='intp')
    tm.assert_numpy_array_equal(result_indexer, expected_indexer)
    tm.assert_numpy_array_equal(result_missing, expected_missing)
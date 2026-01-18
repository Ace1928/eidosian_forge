import re
import numpy as np
import pytest
from pandas.errors import InvalidIndexError
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('size', [1, 5])
def test_get_indexer_length_one_interval(self, size, closed):
    index = IntervalIndex.from_tuples([(0, 5)], closed=closed)
    result = index.get_indexer([Interval(0, 5, closed)] * size)
    expected = np.array([0] * size, dtype='intp')
    tm.assert_numpy_array_equal(result, expected)
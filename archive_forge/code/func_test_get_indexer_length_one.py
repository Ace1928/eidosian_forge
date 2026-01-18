import re
import numpy as np
import pytest
from pandas.errors import InvalidIndexError
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('item', [[3], np.arange(0.5, 5, 0.5)])
def test_get_indexer_length_one(self, item, closed):
    index = IntervalIndex.from_tuples([(0, 5)], closed=closed)
    result = index.get_indexer(item)
    expected = np.array([0] * len(item), dtype='intp')
    tm.assert_numpy_array_equal(result, expected)
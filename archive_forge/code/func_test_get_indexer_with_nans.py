import re
import numpy as np
import pytest
from pandas.errors import InvalidIndexError
from pandas import (
import pandas._testing as tm
def test_get_indexer_with_nans(self):
    index = IntervalIndex([np.nan, np.nan])
    other = IntervalIndex([np.nan])
    assert not index._index_as_unique
    result = index.get_indexer_for(other)
    expected = np.array([0, 1], dtype=np.intp)
    tm.assert_numpy_array_equal(result, expected)
import re
import numpy as np
import pytest
from pandas.errors import InvalidIndexError
from pandas import (
import pandas._testing as tm
def test_get_indexer_multiindex_with_intervals(self):
    interval_index = IntervalIndex.from_tuples([(2.0, 3.0), (0.0, 1.0), (1.0, 2.0)], name='interval')
    foo_index = Index([1, 2, 3], name='foo')
    multi_index = MultiIndex.from_product([foo_index, interval_index])
    result = multi_index.get_level_values('interval').get_indexer_for([Interval(0.0, 1.0)])
    expected = np.array([1, 4, 7], dtype=np.intp)
    tm.assert_numpy_array_equal(result, expected)
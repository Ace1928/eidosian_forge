import numpy as np
import pytest
from pandas.core.dtypes.dtypes import CategoricalDtype
from pandas import (
import pandas._testing as tm
def test_interval_index_category(self):
    index = IntervalIndex.from_breaks(np.arange(3, dtype='uint64'))
    result = CategoricalIndex(index).dtype.categories
    expected = IntervalIndex.from_arrays([0, 1], [1, 2], dtype='interval[uint64, right]')
    tm.assert_index_equal(result, expected)
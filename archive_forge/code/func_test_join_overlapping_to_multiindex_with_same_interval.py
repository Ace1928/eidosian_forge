import pytest
from pandas import (
import pandas._testing as tm
def test_join_overlapping_to_multiindex_with_same_interval(range_index, interval_index):
    multi_index = MultiIndex.from_product([interval_index, range_index])
    result = interval_index.join(multi_index)
    tm.assert_index_equal(result, multi_index)
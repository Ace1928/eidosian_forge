import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.types import (
def test_difference_sort_incomparable():
    idx = MultiIndex.from_product([[1, pd.Timestamp('2000'), 2], ['a', 'b']])
    other = MultiIndex.from_product([[3, pd.Timestamp('2000'), 4], ['c', 'd']])
    msg = 'sort order is undefined for incomparable objects'
    with tm.assert_produces_warning(RuntimeWarning, match=msg):
        result = idx.difference(other)
    tm.assert_index_equal(result, idx)
    result = idx.difference(other, sort=False)
    tm.assert_index_equal(result, idx)
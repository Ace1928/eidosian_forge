import numpy as np
import pytest
from pandas.core.dtypes.common import ensure_platform_int
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_delete_preserves_rangeindex_list_middle(self):
    idx = RangeIndex(0, 6, 1)
    loc = [1, 2, 3, 4]
    result = idx.delete(loc)
    expected = RangeIndex(0, 6, 5)
    tm.assert_index_equal(result, expected, exact=True)
    result = idx.delete(loc[::-1])
    tm.assert_index_equal(result, expected, exact=True)
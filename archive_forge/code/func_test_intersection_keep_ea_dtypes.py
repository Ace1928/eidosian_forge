import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.types import (
@pytest.mark.parametrize('val', [pd.NA, 100])
def test_intersection_keep_ea_dtypes(val, any_numeric_ea_dtype):
    midx = MultiIndex.from_arrays([Series([1, 2], dtype=any_numeric_ea_dtype), [2, 1]], names=['a', None])
    midx2 = MultiIndex.from_arrays([Series([1, 2, val], dtype=any_numeric_ea_dtype), [1, 1, 3]])
    result = midx.intersection(midx2)
    expected = MultiIndex.from_arrays([Series([2], dtype=any_numeric_ea_dtype), [1]])
    tm.assert_index_equal(result, expected)
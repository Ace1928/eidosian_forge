import re
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import SparseArray
@pytest.mark.parametrize('data1,data2,expected_data', (([[1, 2], [3, 4]], [[0.5, 6], [7, 8]], [[np.nan, 3.0], [np.nan, 4.0], [np.nan, 7.0], [6.0, 8.0]]), ([[1, 2], [3, 4]], [[5, 6], [7, 8]], [[np.nan, 3.0], [np.nan, 4.0], [5, 7], [6, 8]])))
def test_getitem_bool_mask_duplicate_columns_mixed_dtypes(self, data1, data2, expected_data):
    df1 = DataFrame(np.array(data1))
    df2 = DataFrame(np.array(data2))
    df = concat([df1, df2], axis=1)
    result = df[df > 2]
    exdict = {i: np.array(col) for i, col in enumerate(expected_data)}
    expected = DataFrame(exdict).rename(columns={2: 0, 3: 1})
    tm.assert_frame_equal(result, expected)
from datetime import datetime
import numpy as np
import pytest
from pandas.core.dtypes.cast import find_common_type
from pandas.core.dtypes.common import is_dtype_equal
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_combine_first_duplicates_rows_for_nan_index_values():
    df1 = DataFrame({'x': [9, 10, 11]}, index=MultiIndex.from_arrays([[1, 2, 3], [np.nan, 5, 6]], names=['a', 'b']))
    df2 = DataFrame({'y': [12, 13, 14]}, index=MultiIndex.from_arrays([[1, 2, 4], [np.nan, 5, 7]], names=['a', 'b']))
    expected = DataFrame({'x': [9.0, 10.0, 11.0, np.nan], 'y': [12.0, 13.0, np.nan, 14.0]}, index=MultiIndex.from_arrays([[1, 2, 3, 4], [np.nan, 5, 6, 7]], names=['a', 'b']))
    combined = df1.combine_first(df2)
    tm.assert_frame_equal(combined, expected)
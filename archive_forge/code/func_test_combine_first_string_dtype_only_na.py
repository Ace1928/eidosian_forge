from datetime import datetime
import numpy as np
import pytest
from pandas.core.dtypes.cast import find_common_type
from pandas.core.dtypes.common import is_dtype_equal
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_combine_first_string_dtype_only_na(self, nullable_string_dtype):
    df = DataFrame({'a': ['962', '85'], 'b': [pd.NA] * 2}, dtype=nullable_string_dtype)
    df2 = DataFrame({'a': ['85'], 'b': [pd.NA]}, dtype=nullable_string_dtype)
    df.set_index(['a', 'b'], inplace=True)
    df2.set_index(['a', 'b'], inplace=True)
    result = df.combine_first(df2)
    expected = DataFrame({'a': ['962', '85'], 'b': [pd.NA] * 2}, dtype=nullable_string_dtype).set_index(['a', 'b'])
    tm.assert_frame_equal(result, expected)
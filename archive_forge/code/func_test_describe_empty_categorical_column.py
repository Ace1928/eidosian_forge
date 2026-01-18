import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_describe_empty_categorical_column(self):
    df = DataFrame({'empty_col': Categorical([])})
    result = df.describe()
    expected = DataFrame({'empty_col': [0, 0, np.nan, np.nan]}, index=['count', 'unique', 'top', 'freq'], dtype='object')
    tm.assert_frame_equal(result, expected)
    assert np.isnan(result.iloc[2, 0])
    assert np.isnan(result.iloc[3, 0])
import pytest
import pandas as pd
from pandas import DataFrame
import pandas._testing as tm
def test_assert_frame_equal_columns_mixed_dtype():
    df = DataFrame([[0, 1, 2]], columns=['foo', 'bar', 42], index=[1, 'test', 2])
    tm.assert_frame_equal(df, df, check_like=True)
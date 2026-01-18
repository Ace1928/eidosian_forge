import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_37477():
    orig = DataFrame({'A': [1, 2, 3], 'B': [3, 4, 5]})
    expected = DataFrame({'A': [1, 2, 3], 'B': [3, 1.2, 5]})
    df = orig.copy()
    with tm.assert_produces_warning(FutureWarning, match='Setting an item of incompatible dtype'):
        df.at[1, 'B'] = 1.2
    tm.assert_frame_equal(df, expected)
    df = orig.copy()
    with tm.assert_produces_warning(FutureWarning, match='Setting an item of incompatible dtype'):
        df.loc[1, 'B'] = 1.2
    tm.assert_frame_equal(df, expected)
    df = orig.copy()
    with tm.assert_produces_warning(FutureWarning, match='Setting an item of incompatible dtype'):
        df.iat[1, 1] = 1.2
    tm.assert_frame_equal(df, expected)
    df = orig.copy()
    with tm.assert_produces_warning(FutureWarning, match='Setting an item of incompatible dtype'):
        df.iloc[1, 1] = 1.2
    tm.assert_frame_equal(df, expected)
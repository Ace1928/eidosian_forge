import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_corr_item_cache(self, using_copy_on_write, warn_copy_on_write):
    df = DataFrame({'A': range(10)})
    df['B'] = range(10)[::-1]
    ser = df['A']
    assert len(df._mgr.arrays) == 2
    _ = df.corr(numeric_only=True)
    if using_copy_on_write:
        ser.iloc[0] = 99
        assert df.loc[0, 'A'] == 0
    else:
        ser.values[0] = 99
        assert df.loc[0, 'A'] == 99
        if not warn_copy_on_write:
            assert df['A'] is ser
        assert df.values[0, 0] == 99
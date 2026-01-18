import numpy as np
import pytest
from pandas.errors import SettingWithCopyWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array
def test_count_read_only_array():
    df = DataFrame({'a': [1, 2], 'b': 3})
    result = df.count()
    result.iloc[0] = 100
    expected = Series([100, 2], index=['a', 'b'])
    tm.assert_series_equal(result, expected)
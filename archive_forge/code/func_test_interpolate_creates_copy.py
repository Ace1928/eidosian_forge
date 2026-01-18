import numpy as np
import pytest
from pandas.errors import SettingWithCopyWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array
def test_interpolate_creates_copy(using_copy_on_write, warn_copy_on_write):
    df = DataFrame({'a': [1.5, np.nan, 3]})
    view = df[:]
    expected = df.copy()
    with tm.assert_cow_warning(warn_copy_on_write):
        df.ffill(inplace=True)
    with tm.assert_cow_warning(warn_copy_on_write):
        df.iloc[0, 0] = 100.5
    if using_copy_on_write:
        tm.assert_frame_equal(view, expected)
    else:
        expected = DataFrame({'a': [100.5, 1.5, 3]})
        tm.assert_frame_equal(view, expected)
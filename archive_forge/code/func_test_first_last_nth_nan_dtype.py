import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_first_last_nth_nan_dtype():
    df = DataFrame({'data': ['A'], 'nans': Series([np.nan], dtype=object)})
    grouped = df.groupby('data')
    expected = df.set_index('data').nans
    tm.assert_series_equal(grouped.nans.first(), expected)
    tm.assert_series_equal(grouped.nans.last(), expected)
    expected = df.nans
    tm.assert_series_equal(grouped.nans.nth(-1), expected)
    tm.assert_series_equal(grouped.nans.nth(0), expected)
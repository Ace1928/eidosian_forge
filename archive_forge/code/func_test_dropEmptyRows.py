import datetime
import dateutil
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_dropEmptyRows(self, float_frame):
    N = len(float_frame.index)
    mat = np.random.default_rng(2).standard_normal(N)
    mat[:5] = np.nan
    frame = DataFrame({'foo': mat}, index=float_frame.index)
    original = Series(mat, index=float_frame.index, name='foo')
    expected = original.dropna()
    inplace_frame1, inplace_frame2 = (frame.copy(), frame.copy())
    smaller_frame = frame.dropna(how='all')
    tm.assert_series_equal(frame['foo'], original)
    return_value = inplace_frame1.dropna(how='all', inplace=True)
    tm.assert_series_equal(smaller_frame['foo'], expected)
    tm.assert_series_equal(inplace_frame1['foo'], expected)
    assert return_value is None
    smaller_frame = frame.dropna(how='all', subset=['foo'])
    return_value = inplace_frame2.dropna(how='all', subset=['foo'], inplace=True)
    tm.assert_series_equal(smaller_frame['foo'], expected)
    tm.assert_series_equal(inplace_frame2['foo'], expected)
    assert return_value is None
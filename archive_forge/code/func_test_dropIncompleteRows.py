import datetime
import dateutil
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_dropIncompleteRows(self, float_frame):
    N = len(float_frame.index)
    mat = np.random.default_rng(2).standard_normal(N)
    mat[:5] = np.nan
    frame = DataFrame({'foo': mat}, index=float_frame.index)
    frame['bar'] = 5
    original = Series(mat, index=float_frame.index, name='foo')
    inp_frame1, inp_frame2 = (frame.copy(), frame.copy())
    smaller_frame = frame.dropna()
    tm.assert_series_equal(frame['foo'], original)
    return_value = inp_frame1.dropna(inplace=True)
    exp = Series(mat[5:], index=float_frame.index[5:], name='foo')
    tm.assert_series_equal(smaller_frame['foo'], exp)
    tm.assert_series_equal(inp_frame1['foo'], exp)
    assert return_value is None
    samesize_frame = frame.dropna(subset=['bar'])
    tm.assert_series_equal(frame['foo'], original)
    assert (frame['bar'] == 5).all()
    return_value = inp_frame2.dropna(subset=['bar'], inplace=True)
    tm.assert_index_equal(samesize_frame.index, float_frame.index)
    tm.assert_index_equal(inp_frame2.index, float_frame.index)
    assert return_value is None
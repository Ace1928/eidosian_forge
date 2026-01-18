import re
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import SparseArray
def test_getitem_callable(self, float_frame):
    result = float_frame[lambda x: 'A']
    expected = float_frame.loc[:, 'A']
    tm.assert_series_equal(result, expected)
    result = float_frame[lambda x: ['A', 'B']]
    expected = float_frame.loc[:, ['A', 'B']]
    tm.assert_frame_equal(result, float_frame.loc[:, ['A', 'B']])
    df = float_frame[:3]
    result = df[lambda x: [True, False, True]]
    expected = float_frame.iloc[[0, 2], :]
    tm.assert_frame_equal(result, expected)
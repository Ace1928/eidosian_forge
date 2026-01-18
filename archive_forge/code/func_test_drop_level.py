import re
import numpy as np
import pytest
from pandas.errors import PerformanceWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_drop_level(self, multiindex_dataframe_random_data):
    frame = multiindex_dataframe_random_data
    result = frame.drop(['bar', 'qux'], level='first')
    expected = frame.iloc[[0, 1, 2, 5, 6]]
    tm.assert_frame_equal(result, expected)
    result = frame.drop(['two'], level='second')
    expected = frame.iloc[[0, 2, 3, 6, 7, 9]]
    tm.assert_frame_equal(result, expected)
    result = frame.T.drop(['bar', 'qux'], axis=1, level='first')
    expected = frame.iloc[[0, 1, 2, 5, 6]].T
    tm.assert_frame_equal(result, expected)
    result = frame.T.drop(['two'], axis=1, level='second')
    expected = frame.iloc[[0, 2, 3, 6, 7, 9]].T
    tm.assert_frame_equal(result, expected)
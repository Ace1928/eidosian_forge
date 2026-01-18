import re
import numpy as np
import pytest
from pandas.errors import PerformanceWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_mixed_depth_drop(self):
    arrays = [['a', 'top', 'top', 'routine1', 'routine1', 'routine2'], ['', 'OD', 'OD', 'result1', 'result2', 'result1'], ['', 'wx', 'wy', '', '', '']]
    tuples = sorted(zip(*arrays))
    index = MultiIndex.from_tuples(tuples)
    df = DataFrame(np.random.default_rng(2).standard_normal((4, 6)), columns=index)
    result = df.drop('a', axis=1)
    expected = df.drop([('a', '', '')], axis=1)
    tm.assert_frame_equal(expected, result)
    result = df.drop(['top'], axis=1)
    expected = df.drop([('top', 'OD', 'wx')], axis=1)
    expected = expected.drop([('top', 'OD', 'wy')], axis=1)
    tm.assert_frame_equal(expected, result)
    result = df.drop(('top', 'OD', 'wx'), axis=1)
    expected = df.drop([('top', 'OD', 'wx')], axis=1)
    tm.assert_frame_equal(expected, result)
    expected = df.drop([('top', 'OD', 'wy')], axis=1)
    expected = df.drop('top', axis=1)
    result = df.drop('result1', level=1, axis=1)
    expected = df.drop([('routine1', 'result1', ''), ('routine2', 'result1', '')], axis=1)
    tm.assert_frame_equal(expected, result)
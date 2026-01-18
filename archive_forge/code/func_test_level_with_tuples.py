import datetime
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_level_with_tuples(self):
    index = MultiIndex(levels=[[('foo', 'bar', 0), ('foo', 'baz', 0), ('foo', 'qux', 0)], [0, 1]], codes=[[0, 0, 1, 1, 2, 2], [0, 1, 0, 1, 0, 1]])
    series = Series(np.random.default_rng(2).standard_normal(6), index=index)
    frame = DataFrame(np.random.default_rng(2).standard_normal((6, 4)), index=index)
    result = series['foo', 'bar', 0]
    result2 = series.loc['foo', 'bar', 0]
    expected = series[:2]
    expected.index = expected.index.droplevel(0)
    tm.assert_series_equal(result, expected)
    tm.assert_series_equal(result2, expected)
    with pytest.raises(KeyError, match="^\\(\\('foo', 'bar', 0\\), 2\\)$"):
        series[('foo', 'bar', 0), 2]
    result = frame.loc['foo', 'bar', 0]
    result2 = frame.xs(('foo', 'bar', 0))
    expected = frame[:2]
    expected.index = expected.index.droplevel(0)
    tm.assert_frame_equal(result, expected)
    tm.assert_frame_equal(result2, expected)
    index = MultiIndex(levels=[[('foo', 'bar'), ('foo', 'baz'), ('foo', 'qux')], [0, 1]], codes=[[0, 0, 1, 1, 2, 2], [0, 1, 0, 1, 0, 1]])
    series = Series(np.random.default_rng(2).standard_normal(6), index=index)
    frame = DataFrame(np.random.default_rng(2).standard_normal((6, 4)), index=index)
    result = series['foo', 'bar']
    result2 = series.loc['foo', 'bar']
    expected = series[:2]
    expected.index = expected.index.droplevel(0)
    tm.assert_series_equal(result, expected)
    tm.assert_series_equal(result2, expected)
    result = frame.loc['foo', 'bar']
    result2 = frame.xs(('foo', 'bar'))
    expected = frame[:2]
    expected.index = expected.index.droplevel(0)
    tm.assert_frame_equal(result, expected)
    tm.assert_frame_equal(result2, expected)
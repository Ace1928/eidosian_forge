import pytest
from pandas import (
import pandas._testing as tm
def test_groupby_sample_oversample():
    values = [1] * 10 + [2] * 10
    df = DataFrame({'a': values, 'b': values})
    result = df.groupby('a').sample(frac=2.0, replace=True)
    values = [1] * 20 + [2] * 20
    expected = DataFrame({'a': values, 'b': values}, index=result.index)
    tm.assert_frame_equal(result, expected)
    result = df.groupby('a')['b'].sample(frac=2.0, replace=True)
    expected = Series(values, name='b', index=result.index)
    tm.assert_series_equal(result, expected)
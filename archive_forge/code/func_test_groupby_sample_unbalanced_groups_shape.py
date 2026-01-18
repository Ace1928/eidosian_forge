import pytest
from pandas import (
import pandas._testing as tm
def test_groupby_sample_unbalanced_groups_shape():
    values = [1] * 10 + [2] * 20
    df = DataFrame({'a': values, 'b': values})
    result = df.groupby('a').sample(n=5)
    values = [1] * 5 + [2] * 5
    expected = DataFrame({'a': values, 'b': values}, index=result.index)
    tm.assert_frame_equal(result, expected)
    result = df.groupby('a')['b'].sample(n=5)
    expected = Series(values, name='b', index=result.index)
    tm.assert_series_equal(result, expected)
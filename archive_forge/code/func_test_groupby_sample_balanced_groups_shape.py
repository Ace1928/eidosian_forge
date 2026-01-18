import pytest
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('n, frac', [(2, None), (None, 0.2)])
def test_groupby_sample_balanced_groups_shape(n, frac):
    values = [1] * 10 + [2] * 10
    df = DataFrame({'a': values, 'b': values})
    result = df.groupby('a').sample(n=n, frac=frac)
    values = [1] * 2 + [2] * 2
    expected = DataFrame({'a': values, 'b': values}, index=result.index)
    tm.assert_frame_equal(result, expected)
    result = df.groupby('a')['b'].sample(n=n, frac=frac)
    expected = Series(values, name='b', index=result.index)
    tm.assert_series_equal(result, expected)
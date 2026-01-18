import pytest
from pandas import (
import pandas._testing as tm
def test_groupby_sample_with_empty_inputs():
    df = DataFrame({'a': [], 'b': []})
    groupby_df = df.groupby('a')
    result = groupby_df.sample()
    expected = df
    tm.assert_frame_equal(result, expected)
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas import DataFrame
import pandas._testing as tm
@td.skip_copy_on_write_invalid_test
def test_copy_cache(self):
    df = DataFrame({'a': [1]})
    df['x'] = [0]
    df['a']
    df.copy()
    df['a'].values[0] = -1
    tm.assert_frame_equal(df, DataFrame({'a': [-1], 'x': [0]}))
    df['y'] = [0]
    assert df['a'].values[0] == -1
    tm.assert_frame_equal(df, DataFrame({'a': [-1], 'x': [0], 'y': [0]}))
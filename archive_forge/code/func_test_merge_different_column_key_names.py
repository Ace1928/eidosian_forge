from datetime import (
import re
import numpy as np
import pytest
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.reshape.concat import concat
from pandas.core.reshape.merge import (
def test_merge_different_column_key_names(self):
    left = DataFrame({'lkey': ['foo', 'bar', 'baz', 'foo'], 'value': [1, 2, 3, 4]})
    right = DataFrame({'rkey': ['foo', 'bar', 'qux', 'foo'], 'value': [5, 6, 7, 8]})
    merged = left.merge(right, left_on='lkey', right_on='rkey', how='outer', sort=True)
    exp = Series(['bar', 'baz', 'foo', 'foo', 'foo', 'foo', np.nan], name='lkey')
    tm.assert_series_equal(merged['lkey'], exp)
    exp = Series(['bar', np.nan, 'foo', 'foo', 'foo', 'foo', 'qux'], name='rkey')
    tm.assert_series_equal(merged['rkey'], exp)
    exp = Series([2, 3, 1, 1, 4, 4, np.nan], name='value_x')
    tm.assert_series_equal(merged['value_x'], exp)
    exp = Series([6, np.nan, 5, 8, 5, 8, 7], name='value_y')
    tm.assert_series_equal(merged['value_y'], exp)
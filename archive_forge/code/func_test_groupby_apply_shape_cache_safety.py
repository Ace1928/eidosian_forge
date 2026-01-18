from datetime import (
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.groupby import get_groupby_method_args
def test_groupby_apply_shape_cache_safety():
    df = DataFrame({'A': ['a', 'a', 'b'], 'B': [1, 2, 3], 'C': [4, 6, 5]})
    gb = df.groupby('A')
    result = gb[['B', 'C']].apply(lambda x: x.astype(float).max() - x.min())
    expected = DataFrame({'B': [1.0, 0.0], 'C': [2.0, 0.0]}, index=Index(['a', 'b'], name='A'))
    tm.assert_frame_equal(result, expected)
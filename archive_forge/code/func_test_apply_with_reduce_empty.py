from datetime import datetime
import warnings
import numpy as np
import pytest
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.frame.common import zip_frames
def test_apply_with_reduce_empty():
    empty_frame = DataFrame()
    x = []
    result = empty_frame.apply(x.append, axis=1, result_type='expand')
    tm.assert_frame_equal(result, empty_frame)
    result = empty_frame.apply(x.append, axis=1, result_type='reduce')
    expected = Series([], dtype=np.float64)
    tm.assert_series_equal(result, expected)
    empty_with_cols = DataFrame(columns=['a', 'b', 'c'])
    result = empty_with_cols.apply(x.append, axis=1, result_type='expand')
    tm.assert_frame_equal(result, empty_with_cols)
    result = empty_with_cols.apply(x.append, axis=1, result_type='reduce')
    expected = Series([], dtype=np.float64)
    tm.assert_series_equal(result, expected)
    assert x == []
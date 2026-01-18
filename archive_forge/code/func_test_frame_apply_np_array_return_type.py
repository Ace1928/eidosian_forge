from datetime import timedelta
import numpy as np
import pytest
from pandas.core.dtypes.dtypes import DatetimeTZDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_frame_apply_np_array_return_type(self, using_infer_string):
    df = DataFrame([['foo']])
    result = df.apply(lambda col: np.array('bar'))
    if using_infer_string:
        expected = Series([np.array(['bar'])])
    else:
        expected = Series(['bar'])
    tm.assert_series_equal(result, expected)
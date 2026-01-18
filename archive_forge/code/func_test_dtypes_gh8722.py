from datetime import timedelta
import numpy as np
import pytest
from pandas.core.dtypes.dtypes import DatetimeTZDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_dtypes_gh8722(self, float_string_frame):
    float_string_frame['bool'] = float_string_frame['A'] > 0
    result = float_string_frame.dtypes
    expected = Series({k: v.dtype for k, v in float_string_frame.items()}, index=result.index)
    tm.assert_series_equal(result, expected)
    msg = 'use_inf_as_na option is deprecated'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        with option_context('use_inf_as_na', True):
            df = DataFrame([[1]])
            result = df.dtypes
            tm.assert_series_equal(result, Series({0: np.dtype('int64')}))
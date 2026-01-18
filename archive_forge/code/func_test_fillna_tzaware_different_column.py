import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
from pandas.tests.frame.common import _check_mixed_float
def test_fillna_tzaware_different_column(self):
    df = DataFrame({'A': date_range('20130101', periods=4, tz='US/Eastern'), 'B': [1, 2, np.nan, np.nan]})
    msg = "DataFrame.fillna with 'method' is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = df.fillna(method='pad')
    expected = DataFrame({'A': date_range('20130101', periods=4, tz='US/Eastern'), 'B': [1.0, 2.0, 2.0, 2.0]})
    tm.assert_frame_equal(result, expected)
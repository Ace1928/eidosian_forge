import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
from pandas.tests.frame.common import _check_mixed_float
def test_fillna_tzaware(self):
    df = DataFrame({'A': [Timestamp('2012-11-11 00:00:00+01:00'), NaT]})
    exp = DataFrame({'A': [Timestamp('2012-11-11 00:00:00+01:00'), Timestamp('2012-11-11 00:00:00+01:00')]})
    msg = "DataFrame.fillna with 'method' is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        res = df.fillna(method='pad')
    tm.assert_frame_equal(res, exp)
    df = DataFrame({'A': [NaT, Timestamp('2012-11-11 00:00:00+01:00')]})
    exp = DataFrame({'A': [Timestamp('2012-11-11 00:00:00+01:00'), Timestamp('2012-11-11 00:00:00+01:00')]})
    msg = "DataFrame.fillna with 'method' is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        res = df.fillna(method='bfill')
    tm.assert_frame_equal(res, exp)
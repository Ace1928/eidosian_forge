import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
from pandas.tests.frame.common import _check_mixed_float
def test_fillna_datelike(self):
    df = DataFrame({'Date': [NaT, Timestamp('2014-1-1')], 'Date2': [Timestamp('2013-1-1'), NaT]})
    expected = df.copy()
    expected['Date'] = expected['Date'].fillna(df.loc[df.index[0], 'Date2'])
    result = df.fillna(value={'Date': df['Date2']})
    tm.assert_frame_equal(result, expected)
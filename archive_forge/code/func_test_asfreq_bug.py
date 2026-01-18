from datetime import timedelta
import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.indexes.timedeltas import timedelta_range
def test_asfreq_bug():
    df = DataFrame(data=[1, 3], index=[timedelta(), timedelta(minutes=3)])
    result = df.resample('1min').asfreq()
    expected = DataFrame(data=[1, np.nan, np.nan, 3], index=timedelta_range('0 day', periods=4, freq='1min'))
    tm.assert_frame_equal(result, expected)
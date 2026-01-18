from datetime import timedelta
import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.indexes.timedeltas import timedelta_range
@td.skip_if_no('pyarrow')
def test_arrow_duration_resample():
    idx = pd.Index(timedelta_range('1 day', periods=5), dtype='duration[ns][pyarrow]')
    expected = Series(np.arange(5, dtype=np.float64), index=idx)
    result = expected.resample('1D').mean()
    tm.assert_series_equal(result, expected)
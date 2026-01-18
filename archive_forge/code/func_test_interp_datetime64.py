import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('method', ['nearest', 'pad'])
def test_interp_datetime64(self, method, tz_naive_fixture):
    pytest.importorskip('scipy')
    df = Series([1, np.nan, 3], index=date_range('1/1/2000', periods=3, tz=tz_naive_fixture))
    warn = None if method == 'nearest' else FutureWarning
    msg = 'Series.interpolate with method=pad is deprecated'
    with tm.assert_produces_warning(warn, match=msg):
        result = df.interpolate(method=method)
    if warn is not None:
        alt = df.ffill()
        tm.assert_series_equal(result, alt)
    expected = Series([1.0, 1.0, 3.0], index=date_range('1/1/2000', periods=3, tz=tz_naive_fixture))
    tm.assert_series_equal(result, expected)
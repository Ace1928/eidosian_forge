import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_interp_scipy_basic(self):
    pytest.importorskip('scipy')
    s = Series([1, 3, np.nan, 12, np.nan, 25])
    expected = Series([1.0, 3.0, 7.5, 12.0, 18.5, 25.0])
    result = s.interpolate(method='slinear')
    tm.assert_series_equal(result, expected)
    msg = "The 'downcast' keyword in Series.interpolate is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = s.interpolate(method='slinear', downcast='infer')
    tm.assert_series_equal(result, expected)
    expected = Series([1, 3, 3, 12, 12, 25])
    result = s.interpolate(method='nearest')
    tm.assert_series_equal(result, expected.astype('float'))
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = s.interpolate(method='nearest', downcast='infer')
    tm.assert_series_equal(result, expected)
    expected = Series([1, 3, 3, 12, 12, 25])
    result = s.interpolate(method='zero')
    tm.assert_series_equal(result, expected.astype('float'))
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = s.interpolate(method='zero', downcast='infer')
    tm.assert_series_equal(result, expected)
    expected = Series([1, 3.0, 6.823529, 12.0, 18.058824, 25.0])
    result = s.interpolate(method='quadratic')
    tm.assert_series_equal(result, expected)
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = s.interpolate(method='quadratic', downcast='infer')
    tm.assert_series_equal(result, expected)
    expected = Series([1.0, 3.0, 6.8, 12.0, 18.2, 25.0])
    result = s.interpolate(method='cubic')
    tm.assert_series_equal(result, expected)
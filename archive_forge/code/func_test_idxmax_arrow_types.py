from datetime import timedelta
from decimal import Decimal
import re
from dateutil.tz import tzlocal
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.compat import (
from pandas.compat.numpy import np_version_gt2
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import (
def test_idxmax_arrow_types(self):
    pytest.importorskip('pyarrow')
    df = DataFrame({'a': [2, 3, 1], 'b': [2, 1, 1]}, dtype='int64[pyarrow]')
    result = df.idxmax()
    expected = Series([1, 0], index=['a', 'b'])
    tm.assert_series_equal(result, expected)
    result = df.idxmin()
    expected = Series([2, 1], index=['a', 'b'])
    tm.assert_series_equal(result, expected)
    df = DataFrame({'a': ['b', 'c', 'a']}, dtype='string[pyarrow]')
    result = df.idxmax(numeric_only=False)
    expected = Series([1], index=['a'])
    tm.assert_series_equal(result, expected)
    result = df.idxmin(numeric_only=False)
    expected = Series([2], index=['a'])
    tm.assert_series_equal(result, expected)
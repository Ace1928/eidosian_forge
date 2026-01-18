import datetime
from decimal import Decimal
import operator
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import ops
@pytest.mark.parametrize('dtype', [None, object])
def test_more_na_comparisons(self, dtype):
    left = Series(['a', np.nan, 'c'], dtype=dtype)
    right = Series(['a', np.nan, 'd'], dtype=dtype)
    result = left == right
    expected = Series([True, False, False])
    tm.assert_series_equal(result, expected)
    result = left != right
    expected = Series([False, True, True])
    tm.assert_series_equal(result, expected)
    result = left == np.nan
    expected = Series([False, False, False])
    tm.assert_series_equal(result, expected)
    result = left != np.nan
    expected = Series([True, True, True])
    tm.assert_series_equal(result, expected)
from datetime import datetime
import re
import numpy as np
import pytest
from pandas.errors import PerformanceWarning
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
from pandas.tests.strings import (
@pytest.mark.parametrize('na, expected', [(None, pd.NA), (True, True), (False, False), (0, False), (3, True), (np.nan, pd.NA)])
@pytest.mark.parametrize('regex', [True, False])
def test_contains_na_kwarg_for_nullable_string_dtype(nullable_string_dtype, na, expected, regex):
    values = Series(['a', 'b', 'c', 'a', np.nan], dtype=nullable_string_dtype)
    result = values.str.contains('a', na=na, regex=regex)
    expected = Series([True, False, False, True, expected], dtype='boolean')
    tm.assert_series_equal(result, expected)
from datetime import datetime
import re
import numpy as np
import pytest
from pandas.errors import PerformanceWarning
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
from pandas.tests.strings import (
def test_fullmatch_case_kwarg(any_string_dtype):
    ser = Series(['ab', 'AB', 'abc', 'ABC'], dtype=any_string_dtype)
    expected_dtype = np.bool_ if any_string_dtype in object_pyarrow_numpy else 'boolean'
    expected = Series([True, False, False, False], dtype=expected_dtype)
    result = ser.str.fullmatch('ab', case=True)
    tm.assert_series_equal(result, expected)
    expected = Series([True, True, False, False], dtype=expected_dtype)
    result = ser.str.fullmatch('ab', case=False)
    tm.assert_series_equal(result, expected)
    with tm.maybe_produces_warning(PerformanceWarning, using_pyarrow(any_string_dtype)):
        result = ser.str.fullmatch('ab', flags=re.IGNORECASE)
    tm.assert_series_equal(result, expected)
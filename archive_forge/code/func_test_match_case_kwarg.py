from datetime import datetime
import re
import numpy as np
import pytest
from pandas.errors import PerformanceWarning
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
from pandas.tests.strings import (
def test_match_case_kwarg(any_string_dtype):
    values = Series(['ab', 'AB', 'abc', 'ABC'], dtype=any_string_dtype)
    result = values.str.match('ab', case=False)
    expected_dtype = np.bool_ if any_string_dtype in object_pyarrow_numpy else 'boolean'
    expected = Series([True, True, True, True], dtype=expected_dtype)
    tm.assert_series_equal(result, expected)
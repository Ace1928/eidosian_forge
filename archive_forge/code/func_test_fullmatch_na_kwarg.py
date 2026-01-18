from datetime import datetime
import re
import numpy as np
import pytest
from pandas.errors import PerformanceWarning
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
from pandas.tests.strings import (
def test_fullmatch_na_kwarg(any_string_dtype):
    ser = Series(['fooBAD__barBAD', 'BAD_BADleroybrown', np.nan, 'foo'], dtype=any_string_dtype)
    result = ser.str.fullmatch('.*BAD[_]+.*BAD', na=False)
    expected_dtype = np.bool_ if any_string_dtype in object_pyarrow_numpy else 'boolean'
    expected = Series([True, False, False, False], dtype=expected_dtype)
    tm.assert_series_equal(result, expected)
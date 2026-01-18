from datetime import datetime
import operator
import numpy as np
import pytest
from pandas import (
def test_lower_upper_mixed_object():
    s = Series(['a', np.nan, 'b', True, datetime.today(), 'foo', None, 1, 2.0])
    result = s.str.upper()
    expected = Series(['A', np.nan, 'B', np.nan, np.nan, 'FOO', None, np.nan, np.nan], dtype=object)
    tm.assert_series_equal(result, expected)
    result = s.str.lower()
    expected = Series(['a', np.nan, 'b', np.nan, np.nan, 'foo', None, np.nan, np.nan], dtype=object)
    tm.assert_series_equal(result, expected)
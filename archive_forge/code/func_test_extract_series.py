from datetime import datetime
import re
import numpy as np
import pytest
from pandas.core.dtypes.dtypes import ArrowDtype
from pandas import (
@pytest.mark.parametrize('name', [None, 'series_name'])
def test_extract_series(name, any_string_dtype):
    s = Series(['A1', 'B2', 'C3'], name=name, dtype=any_string_dtype)
    result = s.str.extract('(_)', expand=True)
    expected = DataFrame([np.nan, np.nan, np.nan], dtype=any_string_dtype)
    tm.assert_frame_equal(result, expected)
    result = s.str.extract('(_)(_)', expand=True)
    expected = DataFrame([[np.nan, np.nan], [np.nan, np.nan], [np.nan, np.nan]], dtype=any_string_dtype)
    tm.assert_frame_equal(result, expected)
    result = s.str.extract('([AB])[123]', expand=True)
    expected = DataFrame(['A', 'B', np.nan], dtype=any_string_dtype)
    tm.assert_frame_equal(result, expected)
    result = s.str.extract('([AB])([123])', expand=True)
    expected = DataFrame([['A', '1'], ['B', '2'], [np.nan, np.nan]], dtype=any_string_dtype)
    tm.assert_frame_equal(result, expected)
    result = s.str.extract('(?P<letter>[AB])', expand=True)
    expected = DataFrame({'letter': ['A', 'B', np.nan]}, dtype=any_string_dtype)
    tm.assert_frame_equal(result, expected)
    result = s.str.extract('(?P<letter>[AB])(?P<number>[123])', expand=True)
    expected = DataFrame([['A', '1'], ['B', '2'], [np.nan, np.nan]], columns=['letter', 'number'], dtype=any_string_dtype)
    tm.assert_frame_equal(result, expected)
    result = s.str.extract('([AB])(?P<number>[123])', expand=True)
    expected = DataFrame([['A', '1'], ['B', '2'], [np.nan, np.nan]], columns=[0, 'number'], dtype=any_string_dtype)
    tm.assert_frame_equal(result, expected)
    result = s.str.extract('([AB])(?:[123])', expand=True)
    expected = DataFrame(['A', 'B', np.nan], dtype=any_string_dtype)
    tm.assert_frame_equal(result, expected)
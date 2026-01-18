from datetime import datetime
import re
import numpy as np
import pytest
from pandas.core.dtypes.dtypes import ArrowDtype
from pandas import (
@pytest.mark.parametrize('pat,expected_names', [('(?P<letter>[AB])?(?P<number>[123])', ['letter', 'number']), ('([AB])?(?P<number>[123])', [0, 'number'])])
def test_extractall_column_names(pat, expected_names, any_string_dtype):
    s = Series(['', 'A1', '32'], dtype=any_string_dtype)
    result = s.str.extractall(pat)
    expected = DataFrame([('A', '1'), (np.nan, '3'), (np.nan, '2')], index=MultiIndex.from_tuples([(1, 0), (2, 0), (2, 1)], names=(None, 'match')), columns=expected_names, dtype=any_string_dtype)
    tm.assert_frame_equal(result, expected)
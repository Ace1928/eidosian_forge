import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.errors import ChainedAssignmentError
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
def test_interp_various(self):
    pytest.importorskip('scipy')
    df = DataFrame({'A': [1, 2, np.nan, 4, 5, np.nan, 7], 'C': [1, 2, 3, 5, 8, 13, 21]})
    df = df.set_index('C')
    expected = df.copy()
    result = df.interpolate(method='polynomial', order=1)
    expected.loc[3, 'A'] = 2.66666667
    expected.loc[13, 'A'] = 5.76923076
    tm.assert_frame_equal(result, expected)
    result = df.interpolate(method='cubic')
    expected.loc[3, 'A'] = 2.81547781
    expected.loc[13, 'A'] = 5.52964175
    tm.assert_frame_equal(result, expected)
    result = df.interpolate(method='nearest')
    expected.loc[3, 'A'] = 2
    expected.loc[13, 'A'] = 5
    tm.assert_frame_equal(result, expected, check_dtype=False)
    result = df.interpolate(method='quadratic')
    expected.loc[3, 'A'] = 2.82150771
    expected.loc[13, 'A'] = 6.12648668
    tm.assert_frame_equal(result, expected)
    result = df.interpolate(method='slinear')
    expected.loc[3, 'A'] = 2.66666667
    expected.loc[13, 'A'] = 5.76923077
    tm.assert_frame_equal(result, expected)
    result = df.interpolate(method='zero')
    expected.loc[3, 'A'] = 2.0
    expected.loc[13, 'A'] = 5
    tm.assert_frame_equal(result, expected, check_dtype=False)
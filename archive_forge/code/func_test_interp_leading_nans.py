import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.errors import ChainedAssignmentError
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('check_scipy', [False, pytest.param(True, marks=td.skip_if_no('scipy'))])
def test_interp_leading_nans(self, check_scipy):
    df = DataFrame({'A': [np.nan, np.nan, 0.5, 0.25, 0], 'B': [np.nan, -3, -3.5, np.nan, -4]})
    result = df.interpolate()
    expected = df.copy()
    expected.loc[3, 'B'] = -3.75
    tm.assert_frame_equal(result, expected)
    if check_scipy:
        result = df.interpolate(method='polynomial', order=1)
        tm.assert_frame_equal(result, expected)
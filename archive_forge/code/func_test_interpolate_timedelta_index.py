import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_interpolate_timedelta_index(self, request, interp_methods_ind):
    """
        Tests for non numerical index types  - object, period, timedelta
        Note that all methods except time, index, nearest and values
        are tested here.
        """
    pytest.importorskip('scipy')
    ind = pd.timedelta_range(start=1, periods=4)
    df = pd.DataFrame([0, 1, np.nan, 3], index=ind)
    method, kwargs = interp_methods_ind
    if method in {'cubic', 'zero'}:
        request.applymarker(pytest.mark.xfail(reason=f'{method} interpolation is not supported for TimedeltaIndex'))
    result = df[0].interpolate(method=method, **kwargs)
    expected = Series([0.0, 1.0, 2.0, 3.0], name=0, index=ind)
    tm.assert_series_equal(result, expected)
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_axis_numeric_only_true(self, interp_method, request, using_array_manager):
    interpolation, method = interp_method
    df = DataFrame([[1, 2, 3], ['a', 'b', 4]])
    result = df.quantile(0.5, axis=1, numeric_only=True, interpolation=interpolation, method=method)
    expected = Series([3.0, 4.0], index=[0, 1], name=0.5)
    if interpolation == 'nearest':
        expected = expected.astype(np.int64)
    if method == 'table' and using_array_manager:
        request.applymarker(pytest.mark.xfail(reason='Axis name incorrectly set.'))
    tm.assert_series_equal(result, expected)
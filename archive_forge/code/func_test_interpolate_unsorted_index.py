import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('ascending, expected_values', [(True, [1, 2, 3, 9, 10]), (False, [10, 9, 3, 2, 1])])
def test_interpolate_unsorted_index(self, ascending, expected_values):
    ts = Series(data=[10, 9, np.nan, 2, 1], index=[10, 9, 3, 2, 1])
    result = ts.sort_index(ascending=ascending).interpolate(method='index')
    expected = Series(data=expected_values, index=expected_values, dtype=float)
    tm.assert_series_equal(result, expected)
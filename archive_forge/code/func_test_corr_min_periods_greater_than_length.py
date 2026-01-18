import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('method', ['pearson', 'spearman', 'kendall'])
def test_corr_min_periods_greater_than_length(self, method):
    pytest.importorskip('scipy')
    df = DataFrame({'A': [1, 2], 'B': [1, 2]})
    result = df.corr(method=method, min_periods=3)
    expected = DataFrame({'A': [np.nan, np.nan], 'B': [np.nan, np.nan]}, index=['A', 'B'])
    tm.assert_frame_equal(result, expected)
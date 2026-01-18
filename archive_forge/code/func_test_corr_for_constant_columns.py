import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('length', [2, 20, 200, 2000])
def test_corr_for_constant_columns(self, length):
    df = DataFrame(length * [[0.4, 0.1]], columns=['A', 'B'])
    result = df.corr()
    expected = DataFrame({'A': [np.nan, np.nan], 'B': [np.nan, np.nan]}, index=['A', 'B'])
    tm.assert_frame_equal(result, expected)
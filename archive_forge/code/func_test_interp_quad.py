import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_interp_quad(self):
    pytest.importorskip('scipy')
    sq = Series([1, 4, np.nan, 16], index=[1, 2, 3, 4])
    result = sq.interpolate(method='quadratic')
    expected = Series([1.0, 4.0, 9.0, 16.0], index=[1, 2, 3, 4])
    tm.assert_series_equal(result, expected)
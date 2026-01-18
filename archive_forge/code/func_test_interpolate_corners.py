import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('kwargs', [{}, pytest.param({'method': 'polynomial', 'order': 1}, marks=td.skip_if_no('scipy'))])
def test_interpolate_corners(self, kwargs):
    s = Series([np.nan, np.nan])
    tm.assert_series_equal(s.interpolate(**kwargs), s)
    s = Series([], dtype=object).interpolate()
    tm.assert_series_equal(s.interpolate(**kwargs), s)
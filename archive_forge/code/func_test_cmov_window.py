import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.api.indexers import BaseIndexer
def test_cmov_window(step):
    pytest.importorskip('scipy')
    vals = np.array([6.95, 15.21, 4.72, 9.12, 13.81, 13.49, 16.68, 9.48, 10.63, 14.48])
    result = Series(vals).rolling(5, win_type='boxcar', center=True, step=step).mean()
    expected_values = [np.nan, np.nan, 9.962, 11.27, 11.564, 12.516, 12.818, 12.952, np.nan, np.nan]
    expected = Series(expected_values)[::step]
    tm.assert_series_equal(expected, result)
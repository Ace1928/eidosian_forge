import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.api.indexers import BaseIndexer
def test_cmov_window_regular_linear_range(win_types, step):
    pytest.importorskip('scipy')
    vals = np.array(range(10), dtype=float)
    xp = vals.copy()
    xp[:2] = np.nan
    xp[-2:] = np.nan
    xp = Series(xp)[::step]
    rs = Series(vals).rolling(5, win_type=win_types, center=True, step=step).mean()
    tm.assert_series_equal(xp, rs)
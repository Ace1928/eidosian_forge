import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.api.indexers import BaseIndexer
def test_cmov_window_regular_missing_data(win_types, step):
    pytest.importorskip('scipy')
    vals = np.array([6.95, 15.21, 4.72, 9.12, 13.81, 13.49, 16.68, np.nan, 10.63, 14.48])
    xps = {'bartlett': [np.nan, np.nan, 9.70333, 10.5225, 8.4425, 9.1925, 12.5575, 14.3675, 15.61667, 13.655], 'blackman': [np.nan, np.nan, 9.04582, 11.41536, 7.73345, 9.17869, 12.79607, 14.20036, 15.8706, 13.655], 'barthann': [np.nan, np.nan, 9.70333, 10.5225, 8.4425, 9.1925, 12.5575, 14.3675, 15.61667, 13.655], 'bohman': [np.nan, np.nan, 8.9444, 11.56327, 7.61599, 9.1764, 12.83559, 14.17267, 15.90976, 13.655], 'hamming': [np.nan, np.nan, 9.59321, 10.29694, 8.71384, 9.56348, 12.38009, 14.20565, 15.24694, 13.69758], 'nuttall': [np.nan, np.nan, 8.47693, 12.2821, 7.04618, 9.16786, 13.02671, 14.03673, 16.08759, 13.65553], 'triang': [np.nan, np.nan, 9.33167, 9.76125, 9.28667, 10.34667, 12.00556, 13.82125, 14.49429, 13.765], 'blackmanharris': [np.nan, np.nan, 8.42526, 12.36824, 6.97691, 9.16438, 13.05052, 14.02175, 16.1098, 13.65509]}
    xp = Series(xps[win_types])[::step]
    rs = Series(vals).rolling(5, win_type=win_types, min_periods=3, step=step).mean()
    tm.assert_series_equal(xp, rs)
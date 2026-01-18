import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.api.indexers import BaseIndexer
from pandas.core.groupby.groupby import get_groupby
@pytest.mark.parametrize('min_periods', [5, 4, 3])
def test_groupby_rolling_center_min_periods(self, min_periods):
    df = DataFrame({'group': ['A'] * 10 + ['B'] * 10, 'data': range(20)})
    window_size = 5
    result = df.groupby('group').rolling(window_size, center=True, min_periods=min_periods).mean()
    result = result.reset_index()[['group', 'data']]
    grp_A_mean = [1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 7.5, 8.0]
    grp_B_mean = [x + 10.0 for x in grp_A_mean]
    num_nans = max(0, min_periods - 3)
    nans = [np.nan] * num_nans
    grp_A_expected = nans + grp_A_mean[num_nans:10 - num_nans] + nans
    grp_B_expected = nans + grp_B_mean[num_nans:10 - num_nans] + nans
    expected = DataFrame({'group': ['A'] * 10 + ['B'] * 10, 'data': grp_A_expected + grp_B_expected})
    tm.assert_frame_equal(result, expected)
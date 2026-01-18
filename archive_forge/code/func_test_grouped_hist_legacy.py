import re
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tests.plotting.common import (
def test_grouped_hist_legacy(self):
    from pandas.plotting._matplotlib.hist import _grouped_hist
    rs = np.random.default_rng(10)
    df = DataFrame(rs.standard_normal((10, 1)), columns=['A'])
    df['B'] = to_datetime(rs.integers(812419200000000000, 819331200000000000, size=10, dtype=np.int64))
    df['C'] = rs.integers(0, 4, 10)
    df['D'] = ['X'] * 10
    axes = _grouped_hist(df.A, by=df.C)
    _check_axes_shape(axes, axes_num=4, layout=(2, 2))
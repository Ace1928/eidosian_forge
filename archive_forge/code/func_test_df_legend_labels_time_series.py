import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas import (
from pandas.tests.plotting.common import (
from pandas.util.version import Version
def test_df_legend_labels_time_series(self):
    pytest.importorskip('scipy')
    ind = date_range('1/1/2014', periods=3)
    df = DataFrame(np.random.default_rng(2).standard_normal((3, 3)), columns=['a', 'b', 'c'], index=ind)
    df2 = DataFrame(np.random.default_rng(2).standard_normal((3, 3)), columns=['d', 'e', 'f'], index=ind)
    df3 = DataFrame(np.random.default_rng(2).standard_normal((3, 3)), columns=['g', 'h', 'i'], index=ind)
    ax = df.plot(legend=True, secondary_y='b')
    _check_legend_labels(ax, labels=['a', 'b (right)', 'c'])
    ax = df2.plot(legend=False, ax=ax)
    _check_legend_labels(ax, labels=['a', 'b (right)', 'c'])
    ax = df3.plot(legend=True, ax=ax)
    _check_legend_labels(ax, labels=['a', 'b (right)', 'c', 'g', 'h', 'i'])
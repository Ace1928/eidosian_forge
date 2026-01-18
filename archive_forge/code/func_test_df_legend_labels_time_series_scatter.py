import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas import (
from pandas.tests.plotting.common import (
from pandas.util.version import Version
def test_df_legend_labels_time_series_scatter(self):
    pytest.importorskip('scipy')
    ind = date_range('1/1/2014', periods=3)
    df = DataFrame(np.random.default_rng(2).standard_normal((3, 3)), columns=['a', 'b', 'c'], index=ind)
    df2 = DataFrame(np.random.default_rng(2).standard_normal((3, 3)), columns=['d', 'e', 'f'], index=ind)
    df3 = DataFrame(np.random.default_rng(2).standard_normal((3, 3)), columns=['g', 'h', 'i'], index=ind)
    ax = df.plot.scatter(x='a', y='b', label='data1')
    _check_legend_labels(ax, labels=['data1'])
    ax = df2.plot.scatter(x='d', y='e', legend=False, label='data2', ax=ax)
    _check_legend_labels(ax, labels=['data1'])
    ax = df3.plot.scatter(x='g', y='h', label='data3', ax=ax)
    _check_legend_labels(ax, labels=['data1', 'data3'])
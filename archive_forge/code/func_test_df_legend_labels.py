import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas import (
from pandas.tests.plotting.common import (
from pandas.util.version import Version
@pytest.mark.parametrize('kind', ['line', 'bar', 'barh', 'kde', 'area', 'hist'])
def test_df_legend_labels(self, kind):
    pytest.importorskip('scipy')
    df = DataFrame(np.random.default_rng(2).random((3, 3)), columns=['a', 'b', 'c'])
    df2 = DataFrame(np.random.default_rng(2).random((3, 3)), columns=['d', 'e', 'f'])
    df3 = DataFrame(np.random.default_rng(2).random((3, 3)), columns=['g', 'h', 'i'])
    df4 = DataFrame(np.random.default_rng(2).random((3, 3)), columns=['j', 'k', 'l'])
    ax = df.plot(kind=kind, legend=True)
    _check_legend_labels(ax, labels=df.columns)
    ax = df2.plot(kind=kind, legend=False, ax=ax)
    _check_legend_labels(ax, labels=df.columns)
    ax = df3.plot(kind=kind, legend=True, ax=ax)
    _check_legend_labels(ax, labels=df.columns.union(df3.columns))
    ax = df4.plot(kind=kind, legend='reverse', ax=ax)
    expected = list(df.columns.union(df3.columns)) + list(reversed(df4.columns))
    _check_legend_labels(ax, labels=expected)
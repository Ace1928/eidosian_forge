import numpy as np
import pytest
from pandas import (
from pandas.tests.plotting.common import (
def test_groupby_hist_series_with_legend(self):
    index = Index(15 * ['1'] + 15 * ['2'], name='c')
    df = DataFrame(np.random.default_rng(2).standard_normal((30, 2)), index=index, columns=['a', 'b'])
    g = df.groupby('c')
    for ax in g['a'].hist(legend=True):
        _check_axes_shape(ax, axes_num=1, layout=(1, 1))
        _check_legend_labels(ax, ['1', '2'])
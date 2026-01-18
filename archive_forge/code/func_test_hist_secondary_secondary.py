import re
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tests.plotting.common import (
def test_hist_secondary_secondary(self):
    df = DataFrame(np.random.default_rng(2).standard_normal((30, 4)), columns=list('abcd'))
    _, ax = mpl.pyplot.subplots()
    ax = df['a'].plot.hist(legend=True, secondary_y=True, ax=ax)
    df['b'].plot.hist(ax=ax, legend=True, secondary_y=True)
    _check_legend_labels(ax.left_ax, labels=['a (right)', 'b (right)'])
    assert not ax.left_ax.get_yaxis().get_visible()
    assert ax.get_yaxis().get_visible()
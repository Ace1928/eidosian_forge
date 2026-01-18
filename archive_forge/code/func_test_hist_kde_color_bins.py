import re
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tests.plotting.common import (
def test_hist_kde_color_bins(self, ts):
    pytest.importorskip('scipy')
    _, ax = mpl.pyplot.subplots()
    ax = ts.plot.hist(logy=True, bins=10, color='b', ax=ax)
    _check_ax_scales(ax, yaxis='log')
    assert len(ax.patches) == 10
    _check_colors(ax.patches, facecolors=['b'] * 10)
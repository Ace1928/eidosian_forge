import re
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tests.plotting.common import (
def test_hist_kde_color(self, ts):
    pytest.importorskip('scipy')
    _, ax = mpl.pyplot.subplots()
    ax = ts.plot.kde(logy=True, color='r', ax=ax)
    _check_ax_scales(ax, yaxis='log')
    lines = ax.get_lines()
    assert len(lines) == 1
    _check_colors(lines, ['r'])
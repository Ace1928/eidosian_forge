import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas import (
from pandas.tests.plotting.common import (
from pandas.util.version import Version
def test_missing_marker_multi_plots_on_same_ax(self):
    df = DataFrame(data=[[1, 1, 1, 1], [2, 2, 4, 8]], columns=['x', 'r', 'g', 'b'])
    _, ax = mpl.pyplot.subplots(nrows=1, ncols=3)
    df.plot(x='x', y='r', linewidth=0, marker='o', color='r', ax=ax[0])
    df.plot(x='x', y='g', linewidth=1, marker='x', color='g', ax=ax[0])
    df.plot(x='x', y='b', linewidth=1, marker='o', color='b', ax=ax[0])
    _check_legend_labels(ax[0], labels=['r', 'g', 'b'])
    _check_legend_marker(ax[0], expected_markers=['o', 'x', 'o'])
    df.plot(x='x', y='b', linewidth=1, marker='o', color='b', ax=ax[1])
    df.plot(x='x', y='r', linewidth=0, marker='o', color='r', ax=ax[1])
    df.plot(x='x', y='g', linewidth=1, marker='x', color='g', ax=ax[1])
    _check_legend_labels(ax[1], labels=['b', 'r', 'g'])
    _check_legend_marker(ax[1], expected_markers=['o', 'o', 'x'])
    df.plot(x='x', y='g', linewidth=1, marker='x', color='g', ax=ax[2])
    df.plot(x='x', y='b', linewidth=1, marker='o', color='b', ax=ax[2])
    df.plot(x='x', y='r', linewidth=0, marker='o', color='r', ax=ax[2])
    _check_legend_labels(ax[2], labels=['g', 'b', 'r'])
    _check_legend_marker(ax[2], expected_markers=['x', 'o', 'o'])
import numpy as np
import numpy.testing as nptest
from numpy.testing import assert_equal
import pytest
from scipy import stats
import statsmodels.api as sm
from statsmodels.graphics import gofplots
from statsmodels.graphics.gofplots import (
from statsmodels.graphics.utils import _import_mpl
@pytest.mark.matplotlib
def test_axis_order(close_figures):
    xx = np.random.normal(10, 1, (100,))
    xy = np.random.normal(1, 0.01, (100,))
    fig = qqplot_2samples(xx, xy, 'x', 'y')
    ax = fig.get_axes()[0]
    y_range = np.diff(ax.get_ylim())[0]
    x_range = np.diff(ax.get_xlim())[0]
    assert y_range < x_range
    xx_long = np.random.normal(10, 1, (1000,))
    fig = qqplot_2samples(xx_long, xy, 'x', 'y')
    ax = fig.get_axes()[0]
    y_range = np.diff(ax.get_ylim())[0]
    x_range = np.diff(ax.get_xlim())[0]
    assert y_range < x_range
    xy_long = np.random.normal(1, 0.01, (1000,))
    fig = qqplot_2samples(xx, xy_long, 'x', 'y')
    ax = fig.get_axes()[0]
    y_range = np.diff(ax.get_ylim())[0]
    x_range = np.diff(ax.get_xlim())[0]
    assert x_range < y_range
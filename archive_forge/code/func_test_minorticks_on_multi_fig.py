from contextlib import nullcontext
import itertools
import locale
import logging
import re
from packaging.version import parse as parse_version
import numpy as np
from numpy.testing import assert_almost_equal, assert_array_equal
import pytest
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
def test_minorticks_on_multi_fig():
    """
    Turning on minor gridlines in a multi-Axes Figure
    that contains more than one boxplot and shares the x-axis
    should not raise an exception.
    """
    fig, ax = plt.subplots()
    ax.boxplot(np.arange(10), positions=[0])
    ax.boxplot(np.arange(10), positions=[0])
    ax.boxplot(np.arange(10), positions=[1])
    ax.grid(which='major')
    ax.grid(which='minor')
    ax.minorticks_on()
    fig.draw_without_rendering()
    assert ax.get_xgridlines()
    assert isinstance(ax.xaxis.get_minor_locator(), mpl.ticker.AutoMinorLocator)
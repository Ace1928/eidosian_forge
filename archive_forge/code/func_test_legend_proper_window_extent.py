import collections
import platform
from unittest import mock
import warnings
import numpy as np
from numpy.testing import assert_allclose
import pytest
from matplotlib.testing.decorators import check_figures_equal, image_comparison
from matplotlib.testing._markers import needs_usetex
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.transforms as mtransforms
import matplotlib.collections as mcollections
import matplotlib.lines as mlines
from matplotlib.legend_handler import HandlerTuple
import matplotlib.legend as mlegend
from matplotlib import _api, rc_context
from matplotlib.font_manager import FontProperties
def test_legend_proper_window_extent():
    fig, ax = plt.subplots(dpi=100)
    ax.plot(range(10), label='Aardvark')
    leg = ax.legend()
    x01 = leg.get_window_extent(fig.canvas.get_renderer()).x0
    fig, ax = plt.subplots(dpi=200)
    ax.plot(range(10), label='Aardvark')
    leg = ax.legend()
    x02 = leg.get_window_extent(fig.canvas.get_renderer()).x0
    assert pytest.approx(x01 * 2, 0.1) == x02
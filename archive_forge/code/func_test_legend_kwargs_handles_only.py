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
def test_legend_kwargs_handles_only(self):
    fig, ax = plt.subplots()
    x = np.linspace(0, 1, 11)
    ln1, = ax.plot(x, x, label='x')
    ln2, = ax.plot(x, 2 * x, label='2x')
    ln3, = ax.plot(x, 3 * x, label='3x')
    with mock.patch('matplotlib.legend.Legend') as Legend:
        ax.legend(handles=[ln3, ln2])
    Legend.assert_called_with(ax, [ln3, ln2], ['3x', '2x'])
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
def test_legend_kwargs_handles_labels(self):
    fig, ax = plt.subplots()
    th = np.linspace(0, 2 * np.pi, 1024)
    lns, = ax.plot(th, np.sin(th), label='sin')
    lnc, = ax.plot(th, np.cos(th), label='cos')
    with mock.patch('matplotlib.legend.Legend') as Legend:
        ax.legend(labels=('a', 'b'), handles=(lnc, lns))
    Legend.assert_called_with(ax, (lnc, lns), ('a', 'b'))
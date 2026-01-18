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
@image_comparison(['legend_expand'], remove_text=True)
def test_legend_expand():
    """Test expand mode"""
    legend_modes = [None, 'expand']
    fig, axs = plt.subplots(len(legend_modes), 1)
    x = np.arange(100)
    for ax, mode in zip(axs, legend_modes):
        ax.plot(x, 50 - x, 'o', label='y=1')
        l1 = ax.legend(loc='upper left', mode=mode)
        ax.add_artist(l1)
        ax.plot(x, x - 50, 'o', label='y=-1')
        l2 = ax.legend(loc='right', mode=mode)
        ax.add_artist(l2)
        ax.legend(loc='lower left', mode=mode, ncols=2)
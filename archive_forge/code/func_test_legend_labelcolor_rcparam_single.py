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
@pytest.mark.parametrize('color', ('red', 'none', (0.5, 0.5, 0.5)))
def test_legend_labelcolor_rcparam_single(color):
    fig, ax = plt.subplots()
    ax.plot(np.arange(10), np.arange(10) * 1, label='#1')
    ax.plot(np.arange(10), np.arange(10) * 2, label='#2')
    ax.plot(np.arange(10), np.arange(10) * 3, label='#3')
    mpl.rcParams['legend.labelcolor'] = color
    leg = ax.legend()
    for text in leg.get_texts():
        assert mpl.colors.same_color(text.get_color(), color)
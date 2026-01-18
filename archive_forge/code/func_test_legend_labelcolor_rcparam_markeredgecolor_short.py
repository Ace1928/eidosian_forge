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
def test_legend_labelcolor_rcparam_markeredgecolor_short():
    fig, ax = plt.subplots()
    ax.plot(np.arange(10), np.arange(10) * 1, label='#1', markeredgecolor='r')
    ax.plot(np.arange(10), np.arange(10) * 2, label='#2', markeredgecolor='g')
    ax.plot(np.arange(10), np.arange(10) * 3, label='#3', markeredgecolor='b')
    mpl.rcParams['legend.labelcolor'] = 'mec'
    leg = ax.legend()
    for text, color in zip(leg.get_texts(), ['r', 'g', 'b']):
        assert mpl.colors.same_color(text.get_color(), color)
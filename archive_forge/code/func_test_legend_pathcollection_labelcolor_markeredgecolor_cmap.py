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
def test_legend_pathcollection_labelcolor_markeredgecolor_cmap():
    fig, ax = plt.subplots()
    edgecolors = mpl.cm.viridis(np.random.rand(10))
    ax.scatter(np.arange(10), np.arange(10), label='#1', c=np.arange(10), edgecolor=edgecolors, cmap='Reds')
    leg = ax.legend(labelcolor='markeredgecolor')
    for text, color in zip(leg.get_texts(), ['k']):
        assert mpl.colors.same_color(text.get_color(), color)
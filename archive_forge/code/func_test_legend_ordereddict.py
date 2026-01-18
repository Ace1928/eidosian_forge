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
def test_legend_ordereddict():
    X = np.random.randn(10)
    Y = np.random.randn(10)
    labels = ['a'] * 5 + ['b'] * 5
    colors = ['r'] * 5 + ['g'] * 5
    fig, ax = plt.subplots()
    for x, y, label, color in zip(X, Y, labels, colors):
        ax.scatter(x, y, label=label, c=color)
    handles, labels = ax.get_legend_handles_labels()
    legend = collections.OrderedDict(zip(labels, handles))
    ax.legend(legend.values(), legend.keys(), loc='center left', bbox_to_anchor=(1, 0.5))
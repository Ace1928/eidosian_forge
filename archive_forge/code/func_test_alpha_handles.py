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
def test_alpha_handles():
    x, n, hh = plt.hist([1, 2, 3], alpha=0.25, label='data', color='red')
    legend = plt.legend()
    for lh in legend.legend_handles:
        lh.set_alpha(1.0)
    assert lh.get_facecolor()[:-1] == hh[1].get_facecolor()[:-1]
    assert lh.get_edgecolor()[:-1] == hh[1].get_edgecolor()[:-1]
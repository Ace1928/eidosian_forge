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
@image_comparison(['legend_auto1'], remove_text=True)
def test_legend_auto1():
    """Test automatic legend placement"""
    fig, ax = plt.subplots()
    x = np.arange(100)
    ax.plot(x, 50 - x, 'o', label='y=1')
    ax.plot(x, x - 50, 'o', label='y=-1')
    ax.legend(loc='best')
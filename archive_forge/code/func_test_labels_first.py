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
@image_comparison(['legend_labels_first.png'], remove_text=True)
def test_labels_first():
    fig, ax = plt.subplots()
    ax.plot(np.arange(10), '-o', label=1)
    ax.plot(np.ones(10) * 5, ':x', label='x')
    ax.plot(np.arange(20, 10, -1), 'd', label='diamond')
    ax.legend(loc='best', markerfirst=False)
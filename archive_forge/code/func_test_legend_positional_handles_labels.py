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
def test_legend_positional_handles_labels(self):
    lines = plt.plot(range(10))
    with mock.patch('matplotlib.legend.Legend') as Legend:
        plt.legend(lines, ['hello world'])
    Legend.assert_called_with(plt.gca(), lines, ['hello world'])
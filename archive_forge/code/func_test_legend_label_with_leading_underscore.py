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
def test_legend_label_with_leading_underscore():
    """
    Test that artists with labels starting with an underscore are not added to
    the legend, and that a warning is issued if one tries to add them
    explicitly.
    """
    fig, ax = plt.subplots()
    line, = ax.plot([0, 1], label='_foo')
    with pytest.warns(_api.MatplotlibDeprecationWarning, match='with an underscore'):
        legend = ax.legend(handles=[line])
    assert len(legend.legend_handles) == 0
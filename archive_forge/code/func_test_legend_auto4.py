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
def test_legend_auto4():
    """
    Check that the legend location with automatic placement is the same,
    whatever the histogram type is. Related to issue #9580.
    """
    fig, axs = plt.subplots(ncols=3, figsize=(6.4, 2.4))
    leg_bboxes = []
    for ax, ht in zip(axs.flat, ('bar', 'step', 'stepfilled')):
        ax.set_title(ht)
        ax.hist([0] + 5 * [9], bins=range(10), label='Legend', histtype=ht)
        leg = ax.legend(loc='best')
        fig.canvas.draw()
        leg_bboxes.append(leg.get_window_extent().transformed(ax.transAxes.inverted()))
    assert_allclose(leg_bboxes[1].bounds, leg_bboxes[0].bounds)
    assert_allclose(leg_bboxes[2].bounds, leg_bboxes[0].bounds)
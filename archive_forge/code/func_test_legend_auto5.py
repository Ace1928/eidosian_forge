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
def test_legend_auto5():
    """
    Check that the automatic placement handle a rather complex
    case with non rectangular patch. Related to issue #9580.
    """
    fig, axs = plt.subplots(ncols=2, figsize=(9.6, 4.8))
    leg_bboxes = []
    for ax, loc in zip(axs.flat, ('center', 'best')):
        for _patch in [mpatches.Ellipse(xy=(0.5, 0.9), width=0.8, height=0.2, fc='C1'), mpatches.Polygon(np.array([[0, 1], [0, 0], [1, 0], [1, 1], [0.9, 1.0], [0.9, 0.1], [0.1, 0.1], [0.1, 1.0], [0.1, 1.0]]), fc='C1'), mpatches.Wedge((0.5, 0.5), 0.5, 0, 360, width=0.05, fc='C0')]:
            ax.add_patch(_patch)
        ax.plot([0.1, 0.9], [0.9, 0.9], label='A segment')
        leg = ax.legend(loc=loc)
        fig.canvas.draw()
        leg_bboxes.append(leg.get_window_extent().transformed(ax.transAxes.inverted()))
    assert_allclose(leg_bboxes[1].bounds, leg_bboxes[0].bounds)
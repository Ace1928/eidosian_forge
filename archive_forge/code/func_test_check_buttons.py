import functools
import io
from unittest import mock
import matplotlib as mpl
from matplotlib.backend_bases import MouseEvent
import matplotlib.colors as mcolors
import matplotlib.widgets as widgets
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from matplotlib.testing.decorators import check_figures_equal, image_comparison
from matplotlib.testing.widgets import (click_and_drag, do_event, get_ax,
import numpy as np
from numpy.testing import assert_allclose
import pytest
@check_figures_equal(extensions=['png'])
def test_check_buttons(fig_test, fig_ref):
    widgets.CheckButtons(fig_test.subplots(), ['tea', 'coffee'], [True, True])
    ax = fig_ref.add_subplot(xticks=[], yticks=[])
    ax.scatter([0.15, 0.15], [2 / 3, 1 / 3], marker='s', transform=ax.transAxes, s=(plt.rcParams['font.size'] / 2) ** 2, c=['none', 'none'])
    ax.scatter([0.15, 0.15], [2 / 3, 1 / 3], marker='x', transform=ax.transAxes, s=(plt.rcParams['font.size'] / 2) ** 2, c=['k', 'k'])
    ax.text(0.25, 2 / 3, 'tea', transform=ax.transAxes, va='center')
    ax.text(0.25, 1 / 3, 'coffee', transform=ax.transAxes, va='center')
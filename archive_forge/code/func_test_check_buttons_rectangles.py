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
def test_check_buttons_rectangles(fig_test, fig_ref):
    cb = widgets.CheckButtons(fig_test.subplots(), ['', ''], [False, False])
    with pytest.warns(DeprecationWarning, match='The rectangles attribute was deprecated'):
        cb.rectangles
    ax = fig_ref.add_subplot(xticks=[], yticks=[])
    ys = [2 / 3, 1 / 3]
    dy = 1 / 3
    w, h = (dy / 2, dy / 2)
    rectangles = [Rectangle(xy=(0.05, ys[i] - h / 2), width=w, height=h, edgecolor='black', facecolor='none', transform=ax.transAxes) for i, y in enumerate(ys)]
    for rectangle in rectangles:
        ax.add_patch(rectangle)
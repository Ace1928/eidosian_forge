from itertools import product
import io
import platform
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib import cbook
from matplotlib.backend_bases import MouseEvent
from matplotlib.colors import LogNorm
from matplotlib.patches import Circle, Ellipse
from matplotlib.transforms import Bbox, TransformedBbox
from matplotlib.testing.decorators import (
from mpl_toolkits.axes_grid1 import (
from mpl_toolkits.axes_grid1.anchored_artists import (
from mpl_toolkits.axes_grid1.axes_divider import (
from mpl_toolkits.axes_grid1.axes_rgb import RGBAxes
from mpl_toolkits.axes_grid1.inset_locator import (
import mpl_toolkits.axes_grid1.mpl_axes
import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
def test_divider_append_axes():
    fig, ax = plt.subplots()
    divider = make_axes_locatable(ax)
    axs = {'main': ax, 'top': divider.append_axes('top', 1.2, pad=0.1, sharex=ax), 'bottom': divider.append_axes('bottom', 1.2, pad=0.1, sharex=ax), 'left': divider.append_axes('left', 1.2, pad=0.1, sharey=ax), 'right': divider.append_axes('right', 1.2, pad=0.1, sharey=ax)}
    fig.canvas.draw()
    bboxes = {k: axs[k].get_window_extent() for k in axs}
    dpi = fig.dpi
    assert bboxes['top'].height == pytest.approx(1.2 * dpi)
    assert bboxes['bottom'].height == pytest.approx(1.2 * dpi)
    assert bboxes['left'].width == pytest.approx(1.2 * dpi)
    assert bboxes['right'].width == pytest.approx(1.2 * dpi)
    assert bboxes['top'].y0 - bboxes['main'].y1 == pytest.approx(0.1 * dpi)
    assert bboxes['main'].y0 - bboxes['bottom'].y1 == pytest.approx(0.1 * dpi)
    assert bboxes['main'].x0 - bboxes['left'].x1 == pytest.approx(0.1 * dpi)
    assert bboxes['right'].x0 - bboxes['main'].x1 == pytest.approx(0.1 * dpi)
    assert bboxes['left'].y0 == bboxes['main'].y0 == bboxes['right'].y0
    assert bboxes['left'].y1 == bboxes['main'].y1 == bboxes['right'].y1
    assert bboxes['top'].x0 == bboxes['main'].x0 == bboxes['bottom'].x0
    assert bboxes['top'].x1 == bboxes['main'].x1 == bboxes['bottom'].x1
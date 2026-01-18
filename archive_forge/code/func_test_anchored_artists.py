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
@image_comparison(['anchored_artists.png'], remove_text=True, style='mpl20')
def test_anchored_artists():
    fig, ax = plt.subplots(figsize=(3, 3))
    ada = AnchoredDrawingArea(40, 20, 0, 0, loc='upper right', pad=0.0, frameon=False)
    p1 = Circle((10, 10), 10)
    ada.drawing_area.add_artist(p1)
    p2 = Circle((30, 10), 5, fc='r')
    ada.drawing_area.add_artist(p2)
    ax.add_artist(ada)
    box = AnchoredAuxTransformBox(ax.transData, loc='upper left')
    el = Ellipse((0, 0), width=0.1, height=0.4, angle=30, color='cyan')
    box.drawing_area.add_artist(el)
    ax.add_artist(box)
    with pytest.warns(mpl.MatplotlibDeprecationWarning):
        ae = AnchoredEllipse(ax.transData, width=0.1, height=0.25, angle=-60, loc='lower left', pad=0.5, borderpad=0.4, frameon=True)
    ax.add_artist(ae)
    asb = AnchoredSizeBar(ax.transData, 0.2, '0.2 units', loc='lower right', pad=0.3, borderpad=0.4, sep=4, fill_bar=True, frameon=False, label_top=True, prop={'size': 20}, size_vertical=0.05, color='green')
    ax.add_artist(asb)
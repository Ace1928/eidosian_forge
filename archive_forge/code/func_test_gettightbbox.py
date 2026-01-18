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
def test_gettightbbox():
    fig, ax = plt.subplots(figsize=(8, 6))
    l, = ax.plot([1, 2, 3], [0, 1, 0])
    ax_zoom = zoomed_inset_axes(ax, 4)
    ax_zoom.plot([1, 2, 3], [0, 1, 0])
    mark_inset(ax, ax_zoom, loc1=1, loc2=3, fc='none', ec='0.3')
    remove_ticks_and_titles(fig)
    bbox = fig.get_tightbbox(fig.canvas.get_renderer())
    np.testing.assert_array_almost_equal(bbox.extents, [-17.7, -13.9, 7.2, 5.4])
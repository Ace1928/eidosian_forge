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
def test_inset_axes_complete():
    dpi = 100
    figsize = (6, 5)
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    fig.subplots_adjust(0.1, 0.1, 0.9, 0.9)
    ins = inset_axes(ax, width=2.0, height=2.0, borderpad=0)
    fig.canvas.draw()
    assert_array_almost_equal(ins.get_position().extents, [(0.9 * figsize[0] - 2.0) / figsize[0], (0.9 * figsize[1] - 2.0) / figsize[1], 0.9, 0.9])
    ins = inset_axes(ax, width='40%', height='30%', borderpad=0)
    fig.canvas.draw()
    assert_array_almost_equal(ins.get_position().extents, [0.9 - 0.8 * 0.4, 0.9 - 0.8 * 0.3, 0.9, 0.9])
    ins = inset_axes(ax, width=1.0, height=1.2, bbox_to_anchor=(200, 100), loc=3, borderpad=0)
    fig.canvas.draw()
    assert_array_almost_equal(ins.get_position().extents, [200 / dpi / figsize[0], 100 / dpi / figsize[1], (200 / dpi + 1) / figsize[0], (100 / dpi + 1.2) / figsize[1]])
    ins1 = inset_axes(ax, width='35%', height='60%', loc=3, borderpad=1)
    ins2 = inset_axes(ax, width='100%', height='100%', bbox_to_anchor=(0, 0, 0.35, 0.6), bbox_transform=ax.transAxes, loc=3, borderpad=1)
    fig.canvas.draw()
    assert_array_equal(ins1.get_position().extents, ins2.get_position().extents)
    with pytest.raises(ValueError):
        ins = inset_axes(ax, width='40%', height='30%', bbox_to_anchor=(0.4, 0.5))
    with pytest.warns(UserWarning):
        ins = inset_axes(ax, width='40%', height='30%', bbox_transform=ax.transAxes)
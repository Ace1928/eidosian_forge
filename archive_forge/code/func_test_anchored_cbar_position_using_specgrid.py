import platform
import numpy as np
import pytest
from matplotlib import cm
import matplotlib.colors as mcolors
import matplotlib as mpl
from matplotlib import rc_context
from matplotlib.testing.decorators import image_comparison
import matplotlib.pyplot as plt
from matplotlib.colors import (
from matplotlib.colorbar import Colorbar
from matplotlib.ticker import FixedLocator, LogFormatter, StrMethodFormatter
from matplotlib.testing.decorators import check_figures_equal
def test_anchored_cbar_position_using_specgrid():
    data = np.arange(1200).reshape(30, 40)
    levels = [0, 200, 400, 600, 800, 1000, 1200]
    shrink = 0.5
    anchor_y = 0.3
    fig, ax = plt.subplots()
    cs = ax.contourf(data, levels=levels)
    cbar = plt.colorbar(cs, ax=ax, use_gridspec=True, location='right', anchor=(1, anchor_y), shrink=shrink)
    x0, y0, x1, y1 = ax.get_position().extents
    cx0, cy0, cx1, cy1 = cbar.ax.get_position().extents
    p0 = (y1 - y0) * anchor_y + y0
    np.testing.assert_allclose([cy1, cy0], [y1 * shrink + (1 - shrink) * p0, p0 * (1 - shrink) + y0 * shrink])
    fig, ax = plt.subplots()
    cs = ax.contourf(data, levels=levels)
    cbar = plt.colorbar(cs, ax=ax, use_gridspec=True, location='left', anchor=(1, anchor_y), shrink=shrink)
    x0, y0, x1, y1 = ax.get_position().extents
    cx0, cy0, cx1, cy1 = cbar.ax.get_position().extents
    p0 = (y1 - y0) * anchor_y + y0
    np.testing.assert_allclose([cy1, cy0], [y1 * shrink + (1 - shrink) * p0, p0 * (1 - shrink) + y0 * shrink])
    shrink = 0.5
    anchor_x = 0.3
    fig, ax = plt.subplots()
    cs = ax.contourf(data, levels=levels)
    cbar = plt.colorbar(cs, ax=ax, use_gridspec=True, location='top', anchor=(anchor_x, 1), shrink=shrink)
    x0, y0, x1, y1 = ax.get_position().extents
    cx0, cy0, cx1, cy1 = cbar.ax.get_position().extents
    p0 = (x1 - x0) * anchor_x + x0
    np.testing.assert_allclose([cx1, cx0], [x1 * shrink + (1 - shrink) * p0, p0 * (1 - shrink) + x0 * shrink])
    shrink = 0.5
    anchor_x = 0.3
    fig, ax = plt.subplots()
    cs = ax.contourf(data, levels=levels)
    cbar = plt.colorbar(cs, ax=ax, use_gridspec=True, location='bottom', anchor=(anchor_x, 1), shrink=shrink)
    x0, y0, x1, y1 = ax.get_position().extents
    cx0, cy0, cx1, cy1 = cbar.ax.get_position().extents
    p0 = (x1 - x0) * anchor_x + x0
    np.testing.assert_allclose([cx1, cx0], [x1 * shrink + (1 - shrink) * p0, p0 * (1 - shrink) + x0 * shrink])
import datetime
import platform
import re
from unittest import mock
import contourpy
import numpy as np
from numpy.testing import (
import matplotlib as mpl
from matplotlib import pyplot as plt, rc_context, ticker
from matplotlib.colors import LogNorm, same_color
import matplotlib.patches as mpatches
from matplotlib.testing.decorators import check_figures_equal, image_comparison
import pytest
@pytest.mark.parametrize('split_collections', [False, True])
@image_comparison(baseline_images=['contour_line_start_on_corner_edge'], extensions=['png'], remove_text=True)
def test_contour_line_start_on_corner_edge(split_collections):
    fig, ax = plt.subplots(figsize=(6, 5))
    x, y = np.meshgrid([0, 1, 2, 3, 4], [0, 1, 2])
    z = 1.2 - (x - 2) ** 2 + (y - 1) ** 2
    mask = np.zeros_like(z, dtype=bool)
    mask[1, 1] = mask[1, 3] = True
    z = np.ma.array(z, mask=mask)
    filled = ax.contourf(x, y, z, corner_mask=True)
    cbar = fig.colorbar(filled)
    lines = ax.contour(x, y, z, corner_mask=True, colors='k')
    cbar.add_lines(lines)
    _maybe_split_collections(split_collections)
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
@image_comparison(baseline_images=['contour_manual'], extensions=['png'], remove_text=True, tol=0.89)
def test_contour_manual(split_collections):
    from matplotlib.contour import ContourSet
    fig, ax = plt.subplots(figsize=(4, 4))
    cmap = 'viridis'
    lines0 = [[[2, 0], [1, 2], [1, 3]]]
    lines1 = [[[3, 0], [3, 2]], [[3, 3], [3, 4]]]
    filled01 = [[[0, 0], [0, 4], [1, 3], [1, 2], [2, 0]]]
    filled12 = [[[2, 0], [3, 0], [3, 2], [1, 3], [1, 2]], [[1, 4], [3, 4], [3, 3]]]
    ContourSet(ax, [0, 1, 2], [filled01, filled12], filled=True, cmap=cmap)
    ContourSet(ax, [1, 2], [lines0, lines1], linewidths=3, colors=['r', 'k'])
    segs = [[[4, 0], [7, 0], [7, 3], [4, 3], [4, 0], [5, 1], [5, 2], [6, 2], [6, 1], [5, 1]]]
    kinds = [[1, 2, 2, 2, 79, 1, 2, 2, 2, 79]]
    ContourSet(ax, [2, 3], [segs], [kinds], filled=True, cmap=cmap)
    ContourSet(ax, [2], [segs], [kinds], colors='k', linewidths=3)
    _maybe_split_collections(split_collections)
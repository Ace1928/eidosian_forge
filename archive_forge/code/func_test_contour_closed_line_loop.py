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
@image_comparison(baseline_images=['contour_closed_line_loop'], extensions=['png'], remove_text=True)
def test_contour_closed_line_loop(split_collections):
    z = [[0, 0, 0], [0, 2, 0], [0, 0, 0], [2, 1, 2]]
    fig, ax = plt.subplots(figsize=(2, 2))
    ax.contour(z, [0.5], linewidths=[20], alpha=0.7)
    ax.set_xlim(-0.1, 2.1)
    ax.set_ylim(-0.1, 3.1)
    _maybe_split_collections(split_collections)
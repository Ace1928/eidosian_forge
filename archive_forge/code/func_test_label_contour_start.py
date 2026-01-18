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
def test_label_contour_start():
    _, ax = plt.subplots(dpi=100)
    lats = lons = np.linspace(-np.pi / 2, np.pi / 2, 50)
    lons, lats = np.meshgrid(lons, lats)
    wave = 0.75 * np.sin(2 * lats) ** 8 * np.cos(4 * lons)
    mean = 0.5 * np.cos(2 * lats) * (np.sin(2 * lats) ** 2 + 2)
    data = wave + mean
    cs = ax.contour(lons, lats, data)
    with mock.patch.object(cs, '_split_path_and_get_label_rotation', wraps=cs._split_path_and_get_label_rotation) as mocked_splitter:
        cs.clabel(fontsize=9)
    idxs = [cargs[0][1] for cargs in mocked_splitter.call_args_list]
    assert 0 in idxs
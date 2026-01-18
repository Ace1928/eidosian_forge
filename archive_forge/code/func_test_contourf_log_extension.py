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
@image_comparison(['contour_log_extension.png'], remove_text=True, style='mpl20', tol=1.444)
def test_contourf_log_extension(split_collections):
    plt.rcParams['pcolormesh.snap'] = False
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 5))
    fig.subplots_adjust(left=0.05, right=0.95)
    data_exp = np.linspace(-7.5, 9.5, 1200)
    data = np.power(10, data_exp).reshape(30, 40)
    levels_exp = np.arange(-4.0, 7.0)
    levels = np.power(10.0, levels_exp)
    c1 = ax1.contourf(data, norm=LogNorm(vmin=data.min(), vmax=data.max()))
    c2 = ax2.contourf(data, levels=levels, norm=LogNorm(vmin=levels.min(), vmax=levels.max()), extend='neither')
    c3 = ax3.contourf(data, levels=levels, norm=LogNorm(vmin=levels.min(), vmax=levels.max()), extend='both')
    cb = plt.colorbar(c1, ax=ax1)
    assert cb.ax.get_ylim() == (1e-08, 10000000000.0)
    cb = plt.colorbar(c2, ax=ax2)
    assert_array_almost_equal_nulp(cb.ax.get_ylim(), np.array((0.0001, 1000000.0)))
    cb = plt.colorbar(c3, ax=ax3)
    _maybe_split_collections(split_collections)
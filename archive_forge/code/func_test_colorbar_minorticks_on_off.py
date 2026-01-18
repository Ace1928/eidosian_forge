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
def test_colorbar_minorticks_on_off():
    np.random.seed(seed=12345)
    data = np.random.randn(20, 20)
    with rc_context({'_internal.classic_mode': False}):
        fig, ax = plt.subplots()
        im = ax.pcolormesh(data, vmin=-2.3, vmax=3.3)
        cbar = fig.colorbar(im, extend='both')
        cbar.minorticks_on()
        np.testing.assert_almost_equal(cbar.ax.yaxis.get_minorticklocs(), [-2.2, -1.8, -1.6, -1.4, -1.2, -0.8, -0.6, -0.4, -0.2, 0.2, 0.4, 0.6, 0.8, 1.2, 1.4, 1.6, 1.8, 2.2, 2.4, 2.6, 2.8, 3.2])
        cbar.minorticks_off()
        np.testing.assert_almost_equal(cbar.ax.yaxis.get_minorticklocs(), [])
        im.set_clim(vmin=-1.2, vmax=1.2)
        cbar.minorticks_on()
        np.testing.assert_almost_equal(cbar.ax.yaxis.get_minorticklocs(), [-1.2, -1.1, -0.9, -0.8, -0.7, -0.6, -0.4, -0.3, -0.2, -0.1, 0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 1.1, 1.2])
    data = np.random.uniform(low=1, high=10, size=(20, 20))
    fig, ax = plt.subplots()
    im = ax.pcolormesh(data, norm=LogNorm())
    cbar = fig.colorbar(im)
    fig.canvas.draw()
    default_minorticklocks = cbar.ax.yaxis.get_minorticklocs()
    cbar.minorticks_off()
    np.testing.assert_equal(cbar.ax.yaxis.get_minorticklocs(), [])
    cbar.minorticks_on()
    np.testing.assert_equal(cbar.ax.yaxis.get_minorticklocs(), default_minorticklocks)
    cbar.minorticks_off()
    cbar.set_ticks([3, 5, 7, 9])
    np.testing.assert_equal(cbar.ax.yaxis.get_minorticklocs(), [])
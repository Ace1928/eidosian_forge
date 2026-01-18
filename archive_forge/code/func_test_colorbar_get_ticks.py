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
def test_colorbar_get_ticks():
    plt.figure()
    data = np.arange(1200).reshape(30, 40)
    levels = [0, 200, 400, 600, 800, 1000, 1200]
    plt.contourf(data, levels=levels)
    userTicks = plt.colorbar(ticks=[0, 600, 1200])
    assert userTicks.get_ticks().tolist() == [0, 600, 1200]
    userTicks.set_ticks([600, 700, 800])
    assert userTicks.get_ticks().tolist() == [600, 700, 800]
    defTicks = plt.colorbar(orientation='horizontal')
    np.testing.assert_allclose(defTicks.get_ticks().tolist(), levels)
    fig, ax = plt.subplots()
    x = np.arange(-3.0, 4.001)
    y = np.arange(-4.0, 3.001)
    X, Y = np.meshgrid(x, y)
    Z = X * Y
    Z = Z[:-1, :-1]
    pcm = ax.pcolormesh(X, Y, Z)
    cbar = fig.colorbar(pcm, ax=ax, extend='both', orientation='vertical')
    ticks = cbar.get_ticks()
    np.testing.assert_allclose(ticks, np.arange(-15, 16, 5))
    assert len(cbar.get_ticks(minor=True)) == 0
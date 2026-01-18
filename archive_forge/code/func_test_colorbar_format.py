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
@pytest.mark.parametrize('fmt', ['%4.2e', '{x:.2e}'])
def test_colorbar_format(fmt):
    x, y = np.ogrid[-4:4:31j, -4:4:31j]
    z = 120000 * np.exp(-x ** 2 - y ** 2)
    fig, ax = plt.subplots()
    im = ax.imshow(z)
    cbar = fig.colorbar(im, format=fmt)
    fig.canvas.draw()
    assert cbar.ax.yaxis.get_ticklabels()[4].get_text() == '8.00e+04'
    im.set_clim([4, 200])
    fig.canvas.draw()
    assert cbar.ax.yaxis.get_ticklabels()[4].get_text() == '2.00e+02'
    im.set_norm(LogNorm(vmin=0.1, vmax=10))
    fig.canvas.draw()
    assert cbar.ax.yaxis.get_ticklabels()[0].get_text() == '$\\mathdefault{10^{-2}}$'
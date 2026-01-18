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
@image_comparison(['colorbar_closed_patch.png'], remove_text=True)
def test_colorbar_closed_patch():
    plt.rcParams['pcolormesh.snap'] = False
    fig = plt.figure(figsize=(8, 6))
    ax1 = fig.add_axes([0.05, 0.85, 0.9, 0.1])
    ax2 = fig.add_axes([0.1, 0.65, 0.75, 0.1])
    ax3 = fig.add_axes([0.05, 0.45, 0.9, 0.1])
    ax4 = fig.add_axes([0.05, 0.25, 0.9, 0.1])
    ax5 = fig.add_axes([0.05, 0.05, 0.9, 0.1])
    cmap = mpl.colormaps['RdBu'].resampled(5)
    im = ax1.pcolormesh(np.linspace(0, 10, 16).reshape((4, 4)), cmap=cmap)
    values = np.linspace(0, 10, 5)
    cbar_kw = dict(orientation='horizontal', values=values, ticks=[])
    with rc_context({'axes.linewidth': 16}):
        plt.colorbar(im, cax=ax2, extend='both', extendfrac=0.5, **cbar_kw)
        plt.colorbar(im, cax=ax3, extend='both', **cbar_kw)
        plt.colorbar(im, cax=ax4, extend='both', extendrect=True, **cbar_kw)
        plt.colorbar(im, cax=ax5, extend='neither', **cbar_kw)
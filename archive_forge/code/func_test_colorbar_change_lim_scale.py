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
@image_comparison(['colorbar_change_lim_scale.png'], remove_text=True, style='mpl20')
def test_colorbar_change_lim_scale():
    fig, ax = plt.subplots(1, 2, constrained_layout=True)
    pc = ax[0].pcolormesh(np.arange(100).reshape(10, 10) + 1)
    cb = fig.colorbar(pc, ax=ax[0], extend='both')
    cb.ax.set_yscale('log')
    pc = ax[1].pcolormesh(np.arange(100).reshape(10, 10) + 1)
    cb = fig.colorbar(pc, ax=ax[1], extend='both')
    cb.ax.set_ylim([20, 90])
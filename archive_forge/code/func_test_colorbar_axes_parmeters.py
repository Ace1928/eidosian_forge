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
def test_colorbar_axes_parmeters():
    fig, ax = plt.subplots(2)
    im = ax[0].imshow([[0, 1], [2, 3]])
    fig.colorbar(im, ax=ax)
    fig.colorbar(im, ax=ax[0])
    fig.colorbar(im, ax=[_ax for _ax in ax])
    fig.colorbar(im, ax=(ax[0], ax[1]))
    fig.colorbar(im, ax={i: _ax for i, _ax in enumerate(ax)}.values())
    fig.draw_without_rendering()
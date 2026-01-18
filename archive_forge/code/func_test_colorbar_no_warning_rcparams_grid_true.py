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
def test_colorbar_no_warning_rcparams_grid_true():
    plt.rcParams['axes.grid'] = True
    fig, ax = plt.subplots()
    ax.grid(False)
    im = ax.pcolormesh([0, 1], [0, 1], [[1]])
    fig.colorbar(im)
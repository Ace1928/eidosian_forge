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
@image_comparison(['colorbar_single_scatter.png'], remove_text=True, savefig_kwarg={'dpi': 40})
def test_colorbar_single_scatter():
    plt.figure()
    x = y = [0]
    z = [50]
    cmap = mpl.colormaps['jet'].resampled(16)
    cs = plt.scatter(x, y, z, c=z, cmap=cmap)
    plt.colorbar(cs)
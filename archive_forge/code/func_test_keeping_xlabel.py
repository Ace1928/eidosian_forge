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
@image_comparison(['colorbar_keeping_xlabel.png'], style='mpl20')
def test_keeping_xlabel():
    arr = np.arange(25).reshape((5, 5))
    fig, ax = plt.subplots()
    im = ax.imshow(arr)
    cbar = plt.colorbar(im)
    cbar.ax.set_xlabel('Visible Xlabel')
    cbar.set_label('YLabel')
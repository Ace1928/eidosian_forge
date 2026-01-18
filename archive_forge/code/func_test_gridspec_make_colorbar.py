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
@image_comparison(['cbar_with_subplots_adjust.png'], remove_text=True, savefig_kwarg={'dpi': 40})
def test_gridspec_make_colorbar():
    plt.figure()
    data = np.arange(1200).reshape(30, 40)
    levels = [0, 200, 400, 600, 800, 1000, 1200]
    plt.subplot(121)
    plt.contourf(data, levels=levels)
    plt.colorbar(use_gridspec=True, orientation='vertical')
    plt.subplot(122)
    plt.contourf(data, levels=levels)
    plt.colorbar(use_gridspec=True, orientation='horizontal')
    plt.subplots_adjust(top=0.95, right=0.95, bottom=0.2, hspace=0.25)
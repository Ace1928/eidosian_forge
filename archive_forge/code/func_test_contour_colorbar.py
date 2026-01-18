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
@image_comparison(['contour_colorbar.png'], remove_text=True, tol=0.01 if platform.machine() in ('aarch64', 'ppc64le', 's390x') else 0)
def test_contour_colorbar():
    fig, ax = plt.subplots(figsize=(4, 2))
    data = np.arange(1200).reshape(30, 40) - 500
    levels = np.array([0, 200, 400, 600, 800, 1000, 1200]) - 500
    CS = ax.contour(data, levels=levels, extend='both')
    fig.colorbar(CS, orientation='horizontal', extend='both')
    fig.colorbar(CS, orientation='vertical')
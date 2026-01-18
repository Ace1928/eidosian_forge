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
def test_colorbar_axes_kw():
    plt.figure()
    plt.imshow([[1, 2], [3, 4]])
    plt.colorbar(orientation='horizontal', fraction=0.2, pad=0.2, shrink=0.5, aspect=10, anchor=(0.0, 0.0), panchor=(0.0, 1.0))
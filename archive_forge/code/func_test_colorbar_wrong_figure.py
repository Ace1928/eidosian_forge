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
def test_colorbar_wrong_figure():
    fig_tl = plt.figure(layout='tight')
    fig_cl = plt.figure(layout='constrained')
    im = fig_cl.add_subplot().imshow([[0, 1]])
    fig_tl.colorbar(im)
    fig_tl.draw_without_rendering()
    fig_cl.draw_without_rendering()
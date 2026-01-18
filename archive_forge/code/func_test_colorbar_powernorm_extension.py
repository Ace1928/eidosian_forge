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
def test_colorbar_powernorm_extension():
    f, ax = plt.subplots()
    cb = Colorbar(ax, norm=PowerNorm(gamma=0.5, vmin=0.0, vmax=1.0), orientation='vertical', extend='both')
    assert cb._values[0] >= 0.0
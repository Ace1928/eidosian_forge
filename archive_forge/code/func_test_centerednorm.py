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
def test_centerednorm():
    fig, ax = plt.subplots(figsize=(1, 3))
    norm = mcolors.CenteredNorm()
    mappable = ax.pcolormesh(np.zeros((3, 3)), norm=norm)
    fig.colorbar(mappable)
    assert (norm.vmin, norm.vmax) == (-0.1, 0.1)
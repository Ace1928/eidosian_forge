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
def test_negative_boundarynorm():
    fig, ax = plt.subplots(figsize=(1, 3))
    cmap = mpl.colormaps['viridis']
    clevs = np.arange(-94, -85)
    norm = BoundaryNorm(clevs, cmap.N)
    cb = fig.colorbar(cm.ScalarMappable(cmap=cmap, norm=norm), cax=ax)
    np.testing.assert_allclose(cb.ax.get_ylim(), [clevs[0], clevs[-1]])
    np.testing.assert_allclose(cb.ax.get_yticks(), clevs)
    clevs = np.arange(85, 94)
    norm = BoundaryNorm(clevs, cmap.N)
    cb = fig.colorbar(cm.ScalarMappable(cmap=cmap, norm=norm), cax=ax)
    np.testing.assert_allclose(cb.ax.get_ylim(), [clevs[0], clevs[-1]])
    np.testing.assert_allclose(cb.ax.get_yticks(), clevs)
    clevs = np.arange(-3, 3)
    norm = BoundaryNorm(clevs, cmap.N)
    cb = fig.colorbar(cm.ScalarMappable(cmap=cmap, norm=norm), cax=ax)
    np.testing.assert_allclose(cb.ax.get_ylim(), [clevs[0], clevs[-1]])
    np.testing.assert_allclose(cb.ax.get_yticks(), clevs)
    clevs = np.arange(-8, 1)
    norm = BoundaryNorm(clevs, cmap.N)
    cb = fig.colorbar(cm.ScalarMappable(cmap=cmap, norm=norm), cax=ax)
    np.testing.assert_allclose(cb.ax.get_ylim(), [clevs[0], clevs[-1]])
    np.testing.assert_allclose(cb.ax.get_yticks(), clevs)
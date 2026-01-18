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
def test_inset_colorbar_layout():
    fig, ax = plt.subplots(constrained_layout=True, figsize=(3, 6))
    pc = ax.imshow(np.arange(100).reshape(10, 10))
    cax = ax.inset_axes([1.02, 0.1, 0.03, 0.8])
    cb = fig.colorbar(pc, cax=cax)
    fig.draw_without_rendering()
    np.testing.assert_allclose(cb.ax.get_position().bounds, [0.87, 0.342, 0.0237, 0.315], atol=0.01)
    assert cb.ax in ax.child_axes
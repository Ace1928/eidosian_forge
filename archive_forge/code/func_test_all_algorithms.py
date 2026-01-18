import datetime
import platform
import re
from unittest import mock
import contourpy
import numpy as np
from numpy.testing import (
import matplotlib as mpl
from matplotlib import pyplot as plt, rc_context, ticker
from matplotlib.colors import LogNorm, same_color
import matplotlib.patches as mpatches
from matplotlib.testing.decorators import check_figures_equal, image_comparison
import pytest
@pytest.mark.parametrize('split_collections', [False, True])
@image_comparison(baseline_images=['contour_all_algorithms'], extensions=['png'], remove_text=True, tol=0.06)
def test_all_algorithms(split_collections):
    algorithms = ['mpl2005', 'mpl2014', 'serial', 'threaded']
    rng = np.random.default_rng(2981)
    x, y = np.meshgrid(np.linspace(0.0, 1.0, 10), np.linspace(0.0, 1.0, 6))
    z = np.sin(15 * x) * np.cos(10 * y) + rng.normal(scale=0.5, size=(6, 10))
    mask = np.zeros_like(z, dtype=bool)
    mask[3, 7] = True
    z = np.ma.array(z, mask=mask)
    _, axs = plt.subplots(2, 2)
    for ax, algorithm in zip(axs.ravel(), algorithms):
        ax.contourf(x, y, z, algorithm=algorithm)
        ax.contour(x, y, z, algorithm=algorithm, colors='k')
        ax.set_title(algorithm)
    _maybe_split_collections(split_collections)
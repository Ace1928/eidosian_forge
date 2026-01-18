import functools
import itertools
import platform
import pytest
from mpl_toolkits.mplot3d import Axes3D, axes3d, proj3d, art3d
import matplotlib as mpl
from matplotlib.backend_bases import (MouseButton, MouseEvent,
from matplotlib import cm
from matplotlib import colors as mcolors, patches as mpatch
from matplotlib.testing.decorators import image_comparison, check_figures_equal
from matplotlib.testing.widgets import mock_event
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.patches import Circle, PathPatch
from matplotlib.path import Path
from matplotlib.text import Text
import matplotlib.pyplot as plt
import numpy as np
@pytest.mark.parametrize('extend, levels', [['both', [2, 4, 6]], ['min', [2, 4, 6, 8]], ['max', [0, 2, 4, 6]]])
@check_figures_equal(extensions=['png'])
def test_contourf3d_extend(fig_test, fig_ref, extend, levels):
    X, Y = np.meshgrid(np.arange(-2, 2, 0.25), np.arange(-2, 2, 0.25))
    Z = X ** 2 + Y ** 2
    cmap = mpl.colormaps['viridis'].copy()
    cmap.set_under(cmap(0))
    cmap.set_over(cmap(255))
    kwargs = {'vmin': 1, 'vmax': 7, 'cmap': cmap}
    ax_ref = fig_ref.add_subplot(projection='3d')
    ax_ref.contourf(X, Y, Z, levels=[0, 2, 4, 6, 8], **kwargs)
    ax_test = fig_test.add_subplot(projection='3d')
    ax_test.contourf(X, Y, Z, levels, extend=extend, **kwargs)
    for ax in [ax_ref, ax_test]:
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_zlim(-10, 10)
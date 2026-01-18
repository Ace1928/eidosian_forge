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
def test_colorbar_pos():
    num_plots = 2
    fig, axs = plt.subplots(1, num_plots, figsize=(4, 5), constrained_layout=True, subplot_kw={'projection': '3d'})
    for ax in axs:
        p_tri = ax.plot_trisurf(np.random.randn(5), np.random.randn(5), np.random.randn(5))
    cbar = plt.colorbar(p_tri, ax=axs, orientation='horizontal')
    fig.canvas.draw()
    assert cbar.ax.get_position().extents[1] < 0.2
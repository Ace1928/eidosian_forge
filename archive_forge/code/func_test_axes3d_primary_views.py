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
@mpl3d_image_comparison(['axes3d_primary_views.png'], style='mpl20')
def test_axes3d_primary_views():
    views = [(90, -90, 0), (0, -90, 0), (0, 0, 0), (-90, 90, 0), (0, 90, 0), (0, 180, 0)]
    fig, axs = plt.subplots(2, 3, subplot_kw={'projection': '3d'})
    for i, ax in enumerate(axs.flat):
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_proj_type('ortho')
        ax.view_init(elev=views[i][0], azim=views[i][1], roll=views[i][2])
    plt.tight_layout()
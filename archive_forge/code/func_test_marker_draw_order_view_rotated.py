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
@check_figures_equal(extensions=['png'])
def test_marker_draw_order_view_rotated(fig_test, fig_ref):
    """
    Test that the draw order changes with the direction.

    If we rotate *azim* by 180 degrees and exchange the colors, the plot
    plot should look the same again.
    """
    azim = 130
    x = [-1, 1]
    y = [1, -1]
    z = [0, 0]
    color = ['b', 'y']
    ax = fig_test.add_subplot(projection='3d')
    ax.set_axis_off()
    ax.scatter(x, y, z, s=3500, c=color)
    ax.view_init(elev=0, azim=azim, roll=0)
    ax = fig_ref.add_subplot(projection='3d')
    ax.set_axis_off()
    ax.scatter(x, y, z, s=3500, c=color[::-1])
    ax.view_init(elev=0, azim=azim - 180, roll=0)
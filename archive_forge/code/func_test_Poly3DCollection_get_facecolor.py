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
def test_Poly3DCollection_get_facecolor():
    y, x = np.ogrid[1:10:100j, 1:10:100j]
    z2 = np.cos(x) ** 3 - np.sin(y) ** 2
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    r = ax.plot_surface(x, y, z2, cmap='hot')
    r.get_facecolor()
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
def test_bar3d_colors():
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    for c in ['red', 'green', 'blue', 'yellow']:
        xs = np.arange(len(c))
        ys = np.zeros_like(xs)
        zs = np.zeros_like(ys)
        ax.bar3d(xs, ys, zs, 1, 1, 1, color=c)
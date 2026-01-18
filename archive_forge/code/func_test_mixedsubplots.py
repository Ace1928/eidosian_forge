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
@mpl3d_image_comparison(['mixedsubplot.png'], style='mpl20')
def test_mixedsubplots():

    def f(t):
        return np.cos(2 * np.pi * t) * np.exp(-t)
    t1 = np.arange(0.0, 5.0, 0.1)
    t2 = np.arange(0.0, 5.0, 0.02)
    fig = plt.figure(figsize=plt.figaspect(2.0))
    ax = fig.add_subplot(2, 1, 1)
    ax.plot(t1, f(t1), 'bo', t2, f(t2), 'k--', markerfacecolor='green')
    ax.grid(True)
    ax = fig.add_subplot(2, 1, 2, projection='3d')
    X, Y = np.meshgrid(np.arange(-5, 5, 0.25), np.arange(-5, 5, 0.25))
    R = np.hypot(X, Y)
    Z = np.sin(R)
    ax.plot_surface(X, Y, Z, rcount=40, ccount=40, linewidth=0, antialiased=False)
    ax.set_zlim3d(-1, 1)
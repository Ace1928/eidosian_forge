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
@image_comparison(baseline_images=['computed_zorder'], remove_text=True, extensions=['png'], style='mpl20')
def test_computed_zorder():
    fig = plt.figure()
    ax1 = fig.add_subplot(221, projection='3d')
    ax2 = fig.add_subplot(222, projection='3d')
    ax2.computed_zorder = False
    corners = ((0, 0, 0), (0, 5, 0), (5, 5, 0), (5, 0, 0))
    for ax in (ax1, ax2):
        tri = art3d.Poly3DCollection([corners], facecolors='white', edgecolors='black', zorder=1)
        ax.add_collection3d(tri)
        ax.plot((2, 2), (2, 2), (0, 4), c='red', zorder=2)
        ax.scatter((3, 3), (1, 3), (1, 3), c='red', zorder=10)
        ax.set_xlim((0, 5.0))
        ax.set_ylim((0, 5.0))
        ax.set_zlim((0, 2.5))
    ax3 = fig.add_subplot(223, projection='3d')
    ax4 = fig.add_subplot(224, projection='3d')
    ax4.computed_zorder = False
    dim = 10
    X, Y = np.meshgrid((-dim, dim), (-dim, dim))
    Z = np.zeros((2, 2))
    angle = 0.5
    X2, Y2 = np.meshgrid((-dim, dim), (0, dim))
    Z2 = Y2 * angle
    X3, Y3 = np.meshgrid((-dim, dim), (-dim, 0))
    Z3 = Y3 * angle
    r = 7
    M = 1000
    th = np.linspace(0, 2 * np.pi, M)
    x, y, z = (r * np.cos(th), r * np.sin(th), angle * r * np.sin(th))
    for ax in (ax3, ax4):
        ax.plot_surface(X2, Y3, Z3, color='blue', alpha=0.5, linewidth=0, zorder=-1)
        ax.plot(x[y < 0], y[y < 0], z[y < 0], lw=5, linestyle='--', color='green', zorder=0)
        ax.plot_surface(X, Y, Z, color='red', alpha=0.5, linewidth=0, zorder=1)
        ax.plot(r * np.sin(th), r * np.cos(th), np.zeros(M), lw=5, linestyle='--', color='black', zorder=2)
        ax.plot_surface(X2, Y2, Z2, color='blue', alpha=0.5, linewidth=0, zorder=3)
        ax.plot(x[y > 0], y[y > 0], z[y > 0], lw=5, linestyle='--', color='green', zorder=4)
        ax.view_init(elev=20, azim=-20, roll=0)
        ax.axis('off')
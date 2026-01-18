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
def test_autoscale():
    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
    assert ax.get_zscale() == 'linear'
    ax.margins(x=0, y=0.1, z=0.2)
    ax.plot([0, 1], [0, 1], [0, 1])
    assert ax.get_w_lims() == (0, 1, -0.1, 1.1, -0.2, 1.2)
    ax.autoscale(False)
    ax.set_autoscalez_on(True)
    ax.plot([0, 2], [0, 2], [0, 2])
    assert ax.get_w_lims() == (0, 1, -0.1, 1.1, -0.4, 2.4)
    ax.autoscale(axis='x')
    ax.plot([0, 2], [0, 2], [0, 2])
    assert ax.get_w_lims() == (0, 2, -0.1, 1.1, -0.4, 2.4)
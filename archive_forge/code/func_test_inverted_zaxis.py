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
def test_inverted_zaxis():
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    assert not ax.zaxis_inverted()
    assert ax.get_zlim() == (0, 1)
    assert ax.get_zbound() == (0, 1)
    ax.set_zbound((0, 2))
    assert not ax.zaxis_inverted()
    assert ax.get_zlim() == (0, 2)
    assert ax.get_zbound() == (0, 2)
    ax.invert_zaxis()
    assert ax.zaxis_inverted()
    assert ax.get_zlim() == (2, 0)
    assert ax.get_zbound() == (0, 2)
    ax.set_zbound(upper=1)
    assert ax.zaxis_inverted()
    assert ax.get_zlim() == (1, 0)
    assert ax.get_zbound() == (0, 1)
    ax.set_zbound(lower=2)
    assert ax.zaxis_inverted()
    assert ax.get_zlim() == (2, 1)
    assert ax.get_zbound() == (1, 2)
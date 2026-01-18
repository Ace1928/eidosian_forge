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
def test_axes3d_focal_length_checks():
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    with pytest.raises(ValueError):
        ax.set_proj_type('persp', focal_length=0)
    with pytest.raises(ValueError):
        ax.set_proj_type('ortho', focal_length=1)
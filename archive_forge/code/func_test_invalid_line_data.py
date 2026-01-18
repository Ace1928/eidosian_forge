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
def test_invalid_line_data():
    with pytest.raises(RuntimeError, match='x must be'):
        art3d.Line3D(0, [], [])
    with pytest.raises(RuntimeError, match='y must be'):
        art3d.Line3D([], 0, [])
    with pytest.raises(RuntimeError, match='z must be'):
        art3d.Line3D([], [], 0)
    line = art3d.Line3D([], [], [])
    with pytest.raises(RuntimeError, match='x must be'):
        line.set_data_3d(0, [], [])
    with pytest.raises(RuntimeError, match='y must be'):
        line.set_data_3d([], 0, [])
    with pytest.raises(RuntimeError, match='z must be'):
        line.set_data_3d([], [], 0)
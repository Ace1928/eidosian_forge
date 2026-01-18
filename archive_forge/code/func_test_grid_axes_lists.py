from itertools import product
import io
import platform
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib import cbook
from matplotlib.backend_bases import MouseEvent
from matplotlib.colors import LogNorm
from matplotlib.patches import Circle, Ellipse
from matplotlib.transforms import Bbox, TransformedBbox
from matplotlib.testing.decorators import (
from mpl_toolkits.axes_grid1 import (
from mpl_toolkits.axes_grid1.anchored_artists import (
from mpl_toolkits.axes_grid1.axes_divider import (
from mpl_toolkits.axes_grid1.axes_rgb import RGBAxes
from mpl_toolkits.axes_grid1.inset_locator import (
import mpl_toolkits.axes_grid1.mpl_axes
import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
def test_grid_axes_lists():
    """Test Grid axes_all, axes_row and axes_column relationship."""
    fig = plt.figure()
    grid = Grid(fig, 111, (2, 3), direction='row')
    assert_array_equal(grid, grid.axes_all)
    assert_array_equal(grid.axes_row, np.transpose(grid.axes_column))
    assert_array_equal(grid, np.ravel(grid.axes_row), 'row')
    assert grid.get_geometry() == (2, 3)
    grid = Grid(fig, 111, (2, 3), direction='column')
    assert_array_equal(grid, np.ravel(grid.axes_column), 'column')
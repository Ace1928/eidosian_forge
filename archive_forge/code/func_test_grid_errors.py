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
@pytest.mark.parametrize('rect, ngrids, error, message', (((1, 1), None, TypeError, 'Incorrect rect format'), (111, -1, ValueError, 'ngrids must be positive'), (111, 7, ValueError, 'ngrids must be positive')))
def test_grid_errors(rect, ngrids, error, message):
    fig = plt.figure()
    with pytest.raises(error, match=message):
        Grid(fig, rect, (2, 3), ngrids=ngrids)
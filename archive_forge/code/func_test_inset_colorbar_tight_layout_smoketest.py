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
def test_inset_colorbar_tight_layout_smoketest():
    fig, ax = plt.subplots(1, 1)
    pts = ax.scatter([0, 1], [0, 1], c=[1, 5])
    cax = inset_axes(ax, width='3%', height='70%')
    plt.colorbar(pts, cax=cax)
    with pytest.warns(UserWarning, match='This figure includes Axes'):
        plt.tight_layout()
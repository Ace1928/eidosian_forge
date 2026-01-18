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
@pytest.mark.parametrize('anchor, error, message', ((None, TypeError, 'anchor must be str'), ('CC', ValueError, "'CC' is not a valid value for anchor"), ((1, 1, 1), TypeError, 'anchor must be str')))
def test_divider_errors(anchor, error, message):
    fig = plt.figure()
    with pytest.raises(error, match=message):
        Divider(fig, [0, 0, 1, 1], [Size.Fixed(1)], [Size.Fixed(1)], anchor=anchor)
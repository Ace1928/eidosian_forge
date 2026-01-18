import numpy as np
from numpy.testing import (
import numpy.ma.testutils as matest
import pytest
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.path import Path
from matplotlib.testing.decorators import image_comparison, check_figures_equal
def check_continuity(interpolator, loc, values=None):
    """
        Checks the continuity of interpolator (and its derivatives) near
        location loc. Can check the value at loc itself if *values* is
        provided.

        *interpolator* TriInterpolator
        *loc* location to test (x0, y0)
        *values* (optional) array [z0, dzx0, dzy0] to check the value at *loc*
        """
    n_star = 24
    epsilon = 1e-10
    k = 100.0
    loc_x, loc_y = loc
    star_x = loc_x + epsilon * np.cos(np.linspace(0.0, 2 * np.pi, n_star))
    star_y = loc_y + epsilon * np.sin(np.linspace(0.0, 2 * np.pi, n_star))
    z = interpolator([loc_x], [loc_y])[0]
    dzx, dzy = interpolator.gradient([loc_x], [loc_y])
    if values is not None:
        assert_array_almost_equal(z, values[0])
        assert_array_almost_equal(dzx[0], values[1])
        assert_array_almost_equal(dzy[0], values[2])
    diff_z = interpolator(star_x, star_y) - z
    tab_dzx, tab_dzy = interpolator.gradient(star_x, star_y)
    diff_dzx = tab_dzx - dzx
    diff_dzy = tab_dzy - dzy
    assert_array_less(diff_z, epsilon * k)
    assert_array_less(diff_dzx, epsilon * k)
    assert_array_less(diff_dzy, epsilon * k)
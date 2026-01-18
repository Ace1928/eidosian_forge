import matplotlib.pyplot as plt
import numpy as np
from numpy.testing import assert_array_almost_equal
import pytest
import cartopy.crs as ccrs
from cartopy.tests.conftest import requires_scipy
def test_contour_update_bounds():
    """Test that contour updates the extent"""
    xs, ys = np.meshgrid(np.linspace(0, 360), np.linspace(-80, 80))
    zs = ys ** 2
    ax = plt.axes(projection=ccrs.Orthographic())
    ax.contour(xs, ys, zs, transform=ccrs.PlateCarree())
    plt.draw()
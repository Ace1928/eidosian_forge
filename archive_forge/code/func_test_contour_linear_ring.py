import matplotlib.pyplot as plt
import numpy as np
from numpy.testing import assert_array_almost_equal
import pytest
import cartopy.crs as ccrs
from cartopy.tests.conftest import requires_scipy
@requires_scipy
def test_contour_linear_ring():
    """Test contourf with a section that only has 3 points."""
    from scipy.interpolate import NearestNDInterpolator
    from scipy.signal import convolve2d
    ax = plt.axes([0.01, 0.05, 0.898, 0.85], projection=ccrs.Mercator(), aspect='equal')
    ax.set_extent([-99.6, -89.0, 39.8, 45.5])
    xbnds = ax.get_xlim()
    ybnds = ax.get_ylim()
    ll = ccrs.Geodetic().transform_point(xbnds[0], ybnds[0], ax.projection)
    ul = ccrs.Geodetic().transform_point(xbnds[0], ybnds[1], ax.projection)
    ur = ccrs.Geodetic().transform_point(xbnds[1], ybnds[1], ax.projection)
    lr = ccrs.Geodetic().transform_point(xbnds[1], ybnds[0], ax.projection)
    xi = np.linspace(min(ll[0], ul[0]), max(lr[0], ur[0]), 100)
    yi = np.linspace(min(ll[1], ul[1]), max(ul[1], ur[1]), 100)
    xi, yi = np.meshgrid(xi, yi)
    nn = NearestNDInterpolator((np.arange(-94, -85), np.arange(36, 45)), np.arange(9))
    vals = nn(xi, yi)
    lons = xi
    lats = yi
    window = np.ones((6, 6))
    vals = convolve2d(vals, window / window.sum(), mode='same', boundary='symm')
    ax.contourf(lons, lats, vals, np.arange(9), transform=ccrs.PlateCarree())
    plt.draw()
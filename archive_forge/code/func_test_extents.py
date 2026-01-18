import matplotlib.pyplot as plt
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
import cartopy.crs as ccrs
def test_extents():
    uk = [-12.5, 4, 49, 60]
    uk_crs = ccrs.Geodetic()
    ax = plt.axes(projection=ccrs.PlateCarree(), label='pc')
    ax.set_extent(uk, crs=uk_crs)
    assert_array_almost_equal(ax.viewLim.get_points(), np.array([[-12.5, 49.0], [4.0, 60.0]]))
    ax = plt.axes(projection=ccrs.NorthPolarStereo(), label='npstere')
    ax.set_extent(uk, crs=uk_crs)
    assert_array_almost_equal(ax.viewLim.get_points(), np.array([[-1034046.22566261, -4765889.76601514], [333263.47741164, -3345219.0594531]]))
    ax = plt.axes(projection=ccrs.PlateCarree(), label='pc')
    ax.set_extent([-1034046, 333263, -4765889, -3345219], crs=ccrs.NorthPolarStereo())
    assert_array_almost_equal(ax.viewLim.get_points(), np.array([[-17.17698577, 48.21879707], [5.68924381, 60.54218893]]))
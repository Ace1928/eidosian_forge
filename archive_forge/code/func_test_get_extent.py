import matplotlib.pyplot as plt
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
import cartopy.crs as ccrs
def test_get_extent():
    uk = [-12.5, 4, 49, 60]
    uk_crs = ccrs.PlateCarree()
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent(uk, crs=uk_crs)
    assert_array_almost_equal(ax.get_extent(uk_crs), uk)
    ax = plt.axes(projection=ccrs.Mercator())
    ax.set_extent(uk, crs=uk_crs)
    assert_array_almost_equal(ax.get_extent(uk_crs), uk)
    ax = plt.axes(projection=ccrs.Mercator(min_latitude=uk[2], max_latitude=uk[3]))
    ax.set_extent(uk, crs=uk_crs)
    assert_array_almost_equal(ax.get_extent(uk_crs), uk, decimal=1)
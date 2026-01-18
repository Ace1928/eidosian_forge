import matplotlib.pyplot as plt
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
import cartopy.crs as ccrs
def test_domain_extents():
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent((-180, 180, -90, 90))
    assert_array_equal(ax.viewLim.get_points(), [[-180, -90], [180, 90]])
    ax.set_extent((-180, 180, -90, 90), ccrs.PlateCarree())
    assert_array_equal(ax.viewLim.get_points(), [[-180, -90], [180, 90]])
    ax = plt.axes(projection=ccrs.PlateCarree(90))
    ax.set_extent((-180, 180, -90, 90))
    assert_array_equal(ax.viewLim.get_points(), [[-180, -90], [180, 90]])
    ax.set_extent((-180, 180, -90, 90), ccrs.PlateCarree(90))
    assert_array_equal(ax.viewLim.get_points(), [[-180, -90], [180, 90]])
    ax = plt.axes(projection=ccrs.OSGB(approx=False))
    ax.set_extent((0, 700000.0, 0, 1300000.0), ccrs.OSGB(approx=False))
    assert_array_equal(ax.viewLim.get_points(), [[0, 0], [700000.0, 1300000.0]])
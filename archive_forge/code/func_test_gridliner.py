import io
from unittest import mock
from matplotlib.collections import PolyCollection
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pytest
from shapely.geos import geos_version
import cartopy.crs as ccrs
from cartopy.mpl import _MPL_36
from cartopy.mpl.geoaxes import GeoAxes
from cartopy.mpl.gridliner import (
from cartopy.mpl.ticker import LongitudeFormatter, LongitudeLocator
@pytest.mark.natural_earth
@pytest.mark.mpl_image_compare(filename='gridliner1.png', tolerance=0.73)
def test_gridliner():
    ny, nx = (2, 4)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(nx, ny, 1, projection=ccrs.PlateCarree())
    ax.set_global()
    ax.coastlines(resolution='110m')
    ax.gridlines(linestyle=':')
    ax = fig.add_subplot(nx, ny, 2, projection=ccrs.OSGB(approx=False))
    ax.set_global()
    ax.coastlines(resolution='110m')
    ax.gridlines(linestyle=':')
    ax = fig.add_subplot(nx, ny, 3, projection=ccrs.OSGB(approx=False))
    ax.set_global()
    ax.coastlines(resolution='110m')
    ax.gridlines(ccrs.PlateCarree(), color='blue', linestyle='-')
    ax.gridlines(ccrs.OSGB(approx=False), linestyle=':')
    ax = fig.add_subplot(nx, ny, 4, projection=ccrs.PlateCarree())
    ax.set_global()
    ax.coastlines(resolution='110m')
    ax.gridlines(ccrs.NorthPolarStereo(), alpha=0.5, linewidth=1.5, linestyle='-')
    ax = fig.add_subplot(nx, ny, 5, projection=ccrs.PlateCarree())
    ax.set_global()
    ax.coastlines(resolution='110m')
    osgb = ccrs.OSGB(approx=False)
    ax.set_extent(tuple(osgb.x_limits) + tuple(osgb.y_limits), crs=osgb)
    ax.gridlines(osgb, linestyle=':')
    ax = fig.add_subplot(nx, ny, 6, projection=ccrs.NorthPolarStereo())
    ax.set_global()
    ax.coastlines(resolution='110m')
    ax.gridlines(alpha=0.5, linewidth=1.5, linestyle='-')
    ax = fig.add_subplot(nx, ny, 7, projection=ccrs.NorthPolarStereo())
    ax.set_global()
    ax.coastlines(resolution='110m')
    osgb = ccrs.OSGB(approx=False)
    ax.set_extent(tuple(osgb.x_limits) + tuple(osgb.y_limits), crs=osgb)
    ax.gridlines(osgb, linestyle=':')
    ax = fig.add_subplot(nx, ny, 8, projection=ccrs.Robinson(central_longitude=135))
    ax.set_global()
    ax.coastlines(resolution='110m')
    ax.gridlines(ccrs.PlateCarree(), alpha=0.5, linewidth=1.5, linestyle='-')
    delta = 0.015
    fig.subplots_adjust(left=0 + delta, right=1 - delta, top=1 - delta, bottom=0 + delta)
    return fig
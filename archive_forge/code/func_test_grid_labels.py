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
@pytest.mark.skipif(geos_version == (3, 9, 0), reason='GEOS intersection bug')
@pytest.mark.natural_earth
@pytest.mark.mpl_image_compare(filename='gridliner_labels.png', tolerance=grid_label_tol)
def test_grid_labels():
    fig = plt.figure(figsize=(10, 10))
    crs_pc = ccrs.PlateCarree()
    crs_merc = ccrs.Mercator()
    ax = fig.add_subplot(3, 2, 1, projection=crs_pc)
    ax.coastlines(resolution='110m')
    ax.gridlines(draw_labels=True)
    ax = fig.add_subplot(3, 2, 2, projection=ccrs.PlateCarree(central_longitude=180))
    ax.coastlines(resolution='110m')
    ax.set_title('Known bug')
    gl = ax.gridlines(crs=crs_pc, draw_labels=True)
    gl.top_labels = False
    gl.left_labels = False
    gl.xlines = False
    ax = fig.add_subplot(3, 2, 3, projection=crs_merc)
    ax.coastlines(resolution='110m')
    gl = ax.gridlines(draw_labels=True)
    gl.xlabel_style = gl.ylabel_style = {'size': 9}
    ax = fig.add_subplot(3, 2, 4, projection=crs_pc)
    ax.coastlines(resolution='110m')
    gl = ax.gridlines(crs=crs_pc, linewidth=2, color='gray', alpha=0.5, linestyle=':')
    gl.bottom_labels = True
    gl.right_labels = True
    gl.xlines = False
    gl.xlocator = mticker.FixedLocator([-180, -45, 45, 180])
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 15, 'color': 'gray'}
    gl.xlabel_style = {'color': 'red'}
    gl.xpadding = 10
    gl.ypadding = 15
    fig.canvas.draw()
    assert len([lb for lb in gl.bottom_label_artists if lb.get_visible()]) == 4
    assert len([lb for lb in gl.top_label_artists if lb.get_visible()]) == 0
    assert len([lb for lb in gl.left_label_artists if lb.get_visible()]) == 0
    assert len([lb for lb in gl.right_label_artists if lb.get_visible()]) != 0
    assert len(gl.xline_artists) == 0
    ax = fig.add_subplot(3, 2, 5, projection=crs_pc)
    ax.set_extent([-20, 10.0, 45.0, 70.0])
    ax.coastlines(resolution='110m')
    ax.gridlines(draw_labels=True)
    ax = fig.add_subplot(3, 2, 6, projection=crs_merc)
    ax.set_extent([-20, 10.0, 45.0, 70.0], crs=crs_pc)
    ax.coastlines(resolution='110m')
    gl = ax.gridlines(draw_labels=True)
    gl.rotate_labels = False
    gl.xlabel_style = gl.ylabel_style = {'size': 9}
    fig.subplots_adjust(wspace=0.25, hspace=0.25)
    return fig
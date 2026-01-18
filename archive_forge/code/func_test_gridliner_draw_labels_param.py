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
@pytest.mark.parametrize('draw_labels, result', [(True, {'left': ['40°N'], 'right': ['40°N', '50°N'], 'top': ['70°E', '100°E', '130°E'], 'bottom': ['100°E']}), (False, {'left': [], 'right': [], 'top': [], 'bottom': []}), (['top', 'left'], {'left': ['40°N'], 'right': [], 'top': ['70°E', '100°E', '130°E'], 'bottom': []}), ({'top': 'x', 'right': 'y'}, {'left': [], 'right': ['40°N', '50°N'], 'top': ['70°E', '100°E', '130°E'], 'bottom': []}), ({'left': 'x'}, {'left': ['70°E'], 'right': [], 'top': [], 'bottom': []}), ({'top': 'y'}, {'left': [], 'right': [], 'top': ['50°N'], 'bottom': []})])
def test_gridliner_draw_labels_param(draw_labels, result):
    fig = plt.figure()
    lambert_crs = ccrs.LambertConformal(central_longitude=105)
    ax = fig.add_subplot(projection=lambert_crs)
    ax.set_extent([75, 130, 18, 54], crs=ccrs.PlateCarree())
    gl = ax.gridlines(draw_labels=draw_labels, rotate_labels=False, dms=True, x_inline=False, y_inline=False)
    gl.xlocator = mticker.FixedLocator([70, 100, 130])
    gl.ylocator = mticker.FixedLocator([40, 50])
    fig.canvas.draw()
    res = {}
    for loc in ('left', 'right', 'top', 'bottom'):
        artists = getattr(gl, f'{loc}_label_artists')
        res[loc] = [a.get_text() for a in artists if a.get_visible()]
    assert res == result
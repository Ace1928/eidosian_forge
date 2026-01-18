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
@pytest.mark.mpl_image_compare(filename='gridliner_labels_bbox_style.png', tolerance=grid_label_tol)
def test_gridliner_labels_bbox_style():
    top = 49.3457868
    left = -124.7844079
    right = -66.9513812
    bottom = 24.7433195
    fig = plt.figure(figsize=(6, 3))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.coastlines(resolution='110m')
    ax.set_extent([left, right, bottom, top], crs=ccrs.PlateCarree())
    gl = ax.gridlines(draw_labels=True)
    gl.labels_bbox_style = {'pad': 0, 'visible': True, 'facecolor': 'white', 'edgecolor': 'black', 'boxstyle': 'round, pad=0.2'}
    return fig
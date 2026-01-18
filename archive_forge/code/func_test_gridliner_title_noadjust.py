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
def test_gridliner_title_noadjust():
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_global()
    ax.set_title('foo')
    ax.gridlines(draw_labels=['left', 'right'], ylocs=[-60, 0, 60])
    fig.draw_without_rendering()
    pos = ax.title.get_position()
    ax.set_extent([-180, 180, -60, 60])
    fig.draw_without_rendering()
    assert ax.title.get_position() == pos
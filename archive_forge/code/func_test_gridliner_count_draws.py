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
def test_gridliner_count_draws():
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_global()
    gl = ax.gridlines()
    with mock.patch.object(gl, '_draw_gridliner', return_value=None) as mocked:
        ax.get_tightbbox(renderer=None)
        mocked.assert_called_once()
    with mock.patch.object(gl, '_draw_gridliner', return_value=None) as mocked:
        fig.draw_without_rendering()
        mocked.assert_called_once()
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
@pytest.mark.mpl_image_compare(baseline_dir='baseline_images/mpl/test_mpl_integration', filename='simple_global.png')
def test_gridliner_remove():
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_global()
    ax.coastlines()
    gl = ax.gridlines(draw_labels=True)
    fig.draw_without_rendering()
    gl.remove()
    assert gl not in ax.artists
    return fig
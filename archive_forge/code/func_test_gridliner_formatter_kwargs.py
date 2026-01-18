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
def test_gridliner_formatter_kwargs():
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_extent([-80, -40.0, 10.0, -30.0])
    gl = ax.gridlines(draw_labels=True, dms=False, formatter_kwargs=dict(cardinal_labels={'west': 'O'}))
    fig.canvas.draw()
    labels = [a.get_text() for a in gl.bottom_label_artists if a.get_visible()]
    assert labels == ['75°O', '70°O', '65°O', '60°O', '55°O', '50°O', '45°O']
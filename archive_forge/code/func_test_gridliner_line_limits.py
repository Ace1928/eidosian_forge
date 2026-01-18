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
def test_gridliner_line_limits():
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.NorthPolarStereo())
    ax.set_global()
    xlim, ylim = (125, 75)
    gl = ax.gridlines(xlim=xlim, ylim=ylim)
    fig.canvas.draw_idle()
    paths = gl.xline_artists[0].get_paths() + gl.yline_artists[0].get_paths()
    for path in paths:
        assert (np.min(path.vertices, axis=0) >= (-xlim, -ylim)).all()
        assert (np.max(path.vertices, axis=0) <= (xlim, ylim)).all()
    xlim = (-125, 150)
    ylim = (50, 70)
    gl = ax.gridlines(xlim=xlim, ylim=ylim)
    fig.canvas.draw_idle()
    paths = gl.xline_artists[0].get_paths() + gl.yline_artists[0].get_paths()
    for path in paths:
        assert (np.min(path.vertices, axis=0) >= (xlim[0], ylim[0])).all()
        assert (np.max(path.vertices, axis=0) <= (xlim[1], ylim[1])).all()
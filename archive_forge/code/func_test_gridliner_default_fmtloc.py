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
@pytest.mark.parametrize('proj,gcrs,xloc,xfmt,xloc_expected,xfmt_expected', [(ccrs.PlateCarree(), ccrs.PlateCarree(), [10, 20], None, mticker.FixedLocator, LongitudeFormatter), (ccrs.PlateCarree(), ccrs.Mercator(), [10, 20], None, mticker.FixedLocator, classic_formatter), (ccrs.PlateCarree(), ccrs.PlateCarree(), mticker.MaxNLocator(nbins=9), None, mticker.MaxNLocator, LongitudeFormatter), (ccrs.PlateCarree(), ccrs.Mercator(), mticker.MaxNLocator(nbins=9), None, mticker.MaxNLocator, classic_formatter), (ccrs.PlateCarree(), ccrs.PlateCarree(), None, None, LongitudeLocator, LongitudeFormatter), (ccrs.PlateCarree(), ccrs.Mercator(), None, None, classic_locator.__class__, classic_formatter), (ccrs.PlateCarree(), ccrs.PlateCarree(), None, mticker.StrMethodFormatter('{x}'), LongitudeLocator, mticker.StrMethodFormatter)])
def test_gridliner_default_fmtloc(proj, gcrs, xloc, xfmt, xloc_expected, xfmt_expected):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection=proj)
    gl = ax.gridlines(crs=gcrs, draw_labels=False, xlocs=xloc, xformatter=xfmt)
    assert isinstance(gl.xlocator, xloc_expected)
    assert isinstance(gl.xformatter, xfmt_expected)
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
@pytest.mark.mpl_image_compare(filename='gridliner_labels_title_adjust.png', tolerance=grid_label_tol)
def test_gridliner_title_adjust():
    projs = [ccrs.Mercator(), ccrs.AlbersEqualArea(), ccrs.LambertConformal(), ccrs.Orthographic()]
    plt.rcParams['axes.titley'] = None
    fig = plt.figure(layout='constrained')
    if _MPL_36:
        fig.get_layout_engine().set(h_pad=1 / 8)
    else:
        fig.set_constrained_layout_pads(h_pad=1 / 8)
    for n, proj in enumerate(projs, 1):
        ax = fig.add_subplot(2, 2, n, projection=proj)
        ax.coastlines()
        ax.gridlines(draw_labels=True)
        ax.set_title(proj.__class__.__name__)
    return fig
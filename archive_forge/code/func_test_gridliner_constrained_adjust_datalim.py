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
@pytest.mark.mpl_image_compare(filename='gridliner_constrained_adjust_datalim.png', tolerance=grid_label_tol)
def test_gridliner_constrained_adjust_datalim():
    fig = plt.figure(figsize=(8, 4), layout='constrained')
    ax = fig.add_subplot(projection=ccrs.PlateCarree())
    ax.set_aspect(aspect='equal', adjustable='datalim')
    collection = PolyCollection(verts=[[[0, 0], [1, 0], [1, 1], [0, 1]], [[1, 0], [2, 0], [2, 1], [1, 1]], [[0, 1], [1, 1], [1, 2], [0, 2]], [[1, 1], [2, 1], [2, 2], [1, 2]]], array=[1, 2, 3, 4])
    ax.add_collection(collection)
    fig.colorbar(collection, ax=ax, location='right')
    ax.autoscale()
    ax.gridlines(draw_labels=['bottom', 'left'], linestyle='-')
    return fig
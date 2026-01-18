from unittest import mock
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import numpy as np
import pytest
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.geoaxes import GeoAxes, GeoAxesSubplot, InterProjectionTransform
@pytest.mark.mpl_image_compare(filename='geoaxes_subslice.png')
def test_geoaxes_no_subslice():
    """Test that we do not trigger matplotlib's line subslice optimization."""
    fig, axes = plt.subplots(1, 2, subplot_kw={'projection': ccrs.Mercator()})
    for ax, num_points in zip(axes, [1000, 1001]):
        lats = np.linspace(35, 37, num_points)
        lons = np.linspace(-117, -115, num_points)
        ax.plot(lons, lats, transform=ccrs.PlateCarree())
    return fig
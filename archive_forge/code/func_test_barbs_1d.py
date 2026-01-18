import re
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pytest
import cartopy.crs as ccrs
from cartopy.mpl import _MPL_38
from cartopy.tests.conftest import requires_scipy
@pytest.mark.natural_earth
@pytest.mark.mpl_image_compare(filename='barbs_1d.png')
def test_barbs_1d():
    x = np.array([20.0, 30.0, -17.0, 15.0])
    y = np.array([-1.0, 35.0, 11.0, 40.0])
    u = np.array([23.0, -18.0, 2.0, -11.0])
    v = np.array([5.0, -4.0, 19.0, 11.0])
    plot_extent = [-21, 40, -5, 45]
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(projection=ccrs.PlateCarree())
    ax.set_extent(plot_extent, crs=ccrs.PlateCarree())
    ax.coastlines(resolution='110m')
    ax.barbs(x, y, u, v, transform=ccrs.PlateCarree(), length=8, linewidth=1, color='#7f7f7f')
    return fig
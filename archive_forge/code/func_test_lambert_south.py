import matplotlib.pyplot as plt
import pytest
import cartopy.crs as ccrs
from cartopy.mpl import _MPL_36
@pytest.mark.natural_earth
@pytest.mark.mpl_image_compare(filename='lambert_conformal_south.png')
def test_lambert_south():
    crs = ccrs.LambertConformal(central_longitude=140, cutoff=65, standard_parallels=(-30, -60))
    ax = plt.axes(projection=crs)
    ax.coastlines()
    ax.gridlines()
    return ax.figure
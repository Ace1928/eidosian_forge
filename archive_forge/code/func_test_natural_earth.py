from xml.etree.ElementTree import ParseError
import matplotlib.pyplot as plt
import pytest
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.tests.conftest import _HAS_PYKDTREE_OR_SCIPY
@pytest.mark.filterwarnings('ignore:Downloading')
@pytest.mark.natural_earth
@pytest.mark.mpl_image_compare(filename='natural_earth.png')
def test_natural_earth():
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.OCEAN)
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAKES, alpha=0.5)
    ax.add_feature(cfeature.RIVERS)
    ax.set_xlim((-20, 60))
    ax.set_ylim((-40, 40))
    return ax.figure
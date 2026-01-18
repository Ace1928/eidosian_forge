from xml.etree.ElementTree import ParseError
import matplotlib.pyplot as plt
import pytest
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.tests.conftest import _HAS_PYKDTREE_OR_SCIPY
@pytest.mark.filterwarnings('ignore:Downloading')
@pytest.mark.natural_earth
@pytest.mark.mpl_image_compare(filename='natural_earth_custom.png')
def test_natural_earth_custom():
    ax = plt.axes(projection=ccrs.PlateCarree())
    feature = cfeature.NaturalEarthFeature('physical', 'coastline', '50m', edgecolor='black', facecolor='none')
    ax.add_feature(feature)
    ax.set_xlim((-26, -12))
    ax.set_ylim((58, 72))
    return ax.figure
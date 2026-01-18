import types
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pytest
import shapely.geometry as sgeom
from cartopy import config
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
from cartopy.tests.conftest import _HAS_PYKDTREE_OR_SCIPY
import cartopy.tests.test_img_tiles as ctest_tiles
@pytest.mark.natural_earth
@pytest.mark.mpl_image_compare(filename='imshow_regional_projected.png', tolerance=0.8)
def test_imshow_projected():
    source_proj = ccrs.PlateCarree()
    img_extent = (-120.67660000000001, -106.32104523100001, 13.2301484511245, 30.766899999999502)
    img = plt.imread(REGIONAL_IMG)
    ax = plt.axes(projection=ccrs.LambertConformal())
    ax.set_extent(img_extent, crs=source_proj)
    ax.coastlines(resolution='50m')
    ax.imshow(img, extent=img_extent, transform=source_proj)
    return ax.figure
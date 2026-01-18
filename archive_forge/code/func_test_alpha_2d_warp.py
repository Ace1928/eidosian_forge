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
def test_alpha_2d_warp():
    plt_crs = ccrs.Geostationary(central_longitude=-155.0)
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(1, 1, 1, projection=plt_crs)
    latlon_crs = ccrs.PlateCarree()
    coords = [-162.0, -148.0, 17.5, 23.0]
    ax.set_extent(coords, crs=latlon_crs)
    fake_data = np.zeros([100, 100])
    fake_alphas = np.zeros(fake_data.shape)
    image = ax.imshow(fake_data, extent=coords, transform=latlon_crs, alpha=fake_alphas)
    image_data = image.get_array()
    image_alpha = image.get_alpha()
    assert image_data.shape == image_alpha.shape
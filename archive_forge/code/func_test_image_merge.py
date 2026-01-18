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
@pytest.mark.network
@pytest.mark.mpl_image_compare(filename='image_merge.png', tolerance=0.03)
def test_image_merge():
    tiles = []
    for i in range(1, 4):
        for j in range(0, 3):
            tiles.append((i, j, 2))
    gt = cimgt.GoogleTiles()
    gt._image_url = types.MethodType(ctest_tiles.GOOGLE_IMAGE_URL_REPLACEMENT, gt)
    images_to_merge = []
    for tile in tiles:
        img, extent, origin = gt.get_image(tile)
        img = np.array(img)
        x = np.linspace(extent[0], extent[1], img.shape[1], endpoint=False)
        y = np.linspace(extent[2], extent[3], img.shape[0], endpoint=False)
        images_to_merge.append([img, x, y, origin])
    img, extent, origin = cimgt._merge_tiles(images_to_merge)
    ax = plt.axes(projection=gt.crs)
    ax.set_global()
    ax.coastlines()
    ax.imshow(img, origin=origin, extent=extent, alpha=0.5)
    return ax.figure
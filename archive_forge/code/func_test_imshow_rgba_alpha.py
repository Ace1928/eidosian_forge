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
def test_imshow_rgba_alpha():
    dy, dx = (3, 4)
    ax = plt.axes(projection=ccrs.Orthographic(-120, 45))
    RGBA = np.linspace(0, 255 * 31, dx * dy * 4, dtype=np.uint8).reshape((dy, dx, 4))
    alpha = np.array([0, 85, 170, 255])
    RGBA[:, :, 3] = alpha
    img = ax.imshow(RGBA, transform=ccrs.PlateCarree())
    assert np.all(np.unique(img.get_array().data[:, :, 3]) == alpha)
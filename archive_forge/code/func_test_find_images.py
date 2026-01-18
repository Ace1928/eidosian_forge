import io
from pathlib import Path
import pickle
import shutil
import sys
import warnings
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
from PIL import Image
import pytest
import shapely.geometry as sgeom
from cartopy import config
import cartopy.io.img_nest as cimg_nest
import cartopy.io.img_tiles as cimgt
@pytest.mark.network
def test_find_images(wmts_data):
    z2_dir = _TEST_DATA_DIR / 'z_2'
    img_fname = z2_dir / 'x_2_y_0.png'
    world_file_fname = z2_dir / 'x_2_y_0.pgw'
    img = RoundedImg.from_world_file(img_fname, world_file_fname)
    assert img.filename == img_fname
    assert_array_almost_equal(img.extent, (0.0, 10018754.17139462, 10018754.17139462, 20037508.342789244), decimal=4)
    assert img.origin == 'lower'
    assert_array_equal(img, np.array(Image.open(img.filename)))
    assert img.pixel_size == (39135.7585, 39135.7585)
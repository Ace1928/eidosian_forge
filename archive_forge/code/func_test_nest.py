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
def test_nest(nest_from_config):
    crs = cimgt.GoogleTiles().crs
    z0 = cimg_nest.ImageCollection('aerial z0 test', crs)
    z0.scan_dir_for_imgs(_TEST_DATA_DIR / 'z_0', glob_pattern='*.png', img_class=RoundedImg)
    z1 = cimg_nest.ImageCollection('aerial z1 test', crs)
    z1.scan_dir_for_imgs(_TEST_DATA_DIR / 'z_1', glob_pattern='*.png', img_class=RoundedImg)
    z2 = cimg_nest.ImageCollection('aerial z2 test', crs)
    z2.scan_dir_for_imgs(_TEST_DATA_DIR / 'z_2', glob_pattern='*.png', img_class=RoundedImg)
    for img in z1.images:
        if not z0.images[0].bbox().contains(img.bbox()):
            raise OSError(f"""The test images aren't all "contained" by the z0 images, the nest cannot possibly work.\nimg {img!s} not contained by {z0.images[0]!s}\nExtents: {img.extent!s}; {z0.images[0].extent!s}""")
    nest_z0_z1 = cimg_nest.NestedImageCollection('aerial test', crs, [z0, z1])
    nest = cimg_nest.NestedImageCollection('aerial test', crs, [z0, z1, z2])
    z0_key = ('aerial z0 test', z0.images[0])
    assert z0_key in nest_z0_z1._ancestry.keys()
    assert len(nest_z0_z1._ancestry) == 1
    for img in z1.images:
        key = ('aerial z0 test', z0.images[0])
        assert ('aerial z1 test', img) in nest_z0_z1._ancestry[key]
    x1_y0_z1, = (img for img in z1.images if img.filename.name.endswith('x_1_y_0.png'))
    assert (1, 0, 1) == _tile_from_img(x1_y0_z1)
    assert [(2, 0, 2), (2, 1, 2), (3, 0, 2), (3, 1, 2)] == sorted((_tile_from_img(img) for z, img in nest.subtiles(('aerial z1 test', x1_y0_z1))))
    for name in nest_z0_z1._collections_by_name.keys():
        for img in nest_z0_z1._collections_by_name[name].images:
            collection = nest_from_config._collections_by_name[name]
            assert img in collection.images
    assert nest_z0_z1._ancestry == nest_from_config._ancestry
    s = io.BytesIO()
    pickle.dump(nest_z0_z1, s)
    s.seek(0)
    nest_z0_z1_from_pickle = pickle.load(s)
    assert nest_z0_z1._ancestry == nest_z0_z1_from_pickle._ancestry
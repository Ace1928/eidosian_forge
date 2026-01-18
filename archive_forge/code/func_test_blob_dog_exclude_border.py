import math
import numpy as np
import pytest
from numpy.testing import assert_almost_equal
from skimage import feature
from skimage.draw import disk
from skimage.draw.draw3d import ellipsoid
from skimage.feature import blob_dog, blob_doh, blob_log
from skimage.feature.blob import _blob_overlap
@pytest.mark.parametrize('disc_center', [(5, 5), (5, 20)])
@pytest.mark.parametrize('exclude_border', [6, (6, 6), (4, 15)])
def test_blob_dog_exclude_border(disc_center, exclude_border):
    img = np.ones((512, 512))
    xs, ys = disk(disc_center, 5)
    img[xs, ys] = 255
    blobs = blob_dog(img, min_sigma=1.5, max_sigma=5, sigma_ratio=1.2)
    assert blobs.shape[0] == 1, 'one blob should have been detected'
    b = blobs[0]
    assert b[0] == disc_center[0], f'blob should be {disc_center[0]} px from x border'
    assert b[1] == disc_center[1], f'blob should be {disc_center[1]} px from y border'
    blobs = blob_dog(img, min_sigma=1.5, max_sigma=5, sigma_ratio=1.2, exclude_border=exclude_border)
    if disc_center == (5, 20) and exclude_border == (4, 15):
        assert blobs.shape[0] == 1, 'one blob should have been detected'
        b = blobs[0]
        assert b[0] == disc_center[0], f'blob should be {disc_center[0]} px from x border'
        assert b[1] == disc_center[1], f'blob should be {disc_center[1]} px from y border'
    else:
        msg = 'zero blobs should be detected, as only blob is 5 px from border'
        assert blobs.shape[0] == 0, msg
import math
import numpy as np
import pytest
from numpy.testing import assert_almost_equal
from skimage import feature
from skimage.draw import disk
from skimage.draw.draw3d import ellipsoid
from skimage.feature import blob_dog, blob_doh, blob_log
from skimage.feature.blob import _blob_overlap
def test_blob_doh_log_scale():
    img = np.ones((512, 512), dtype=np.uint8)
    xs, ys = disk((400, 130), 20)
    img[xs, ys] = 255
    xs, ys = disk((460, 50), 30)
    img[xs, ys] = 255
    xs, ys = disk((100, 300), 40)
    img[xs, ys] = 255
    xs, ys = disk((200, 350), 50)
    img[xs, ys] = 255
    blobs = blob_doh(img, min_sigma=1, max_sigma=60, num_sigma=10, log_scale=True, threshold=0.05)

    def radius(x):
        return x[2]
    s = sorted(blobs, key=radius)
    thresh = 10
    b = s[0]
    assert abs(b[0] - 400) <= thresh
    assert abs(b[1] - 130) <= thresh
    assert abs(radius(b) - 20) <= thresh
    b = s[2]
    assert abs(b[0] - 460) <= thresh
    assert abs(b[1] - 50) <= thresh
    assert abs(radius(b) - 30) <= thresh
    b = s[1]
    assert abs(b[0] - 100) <= thresh
    assert abs(b[1] - 300) <= thresh
    assert abs(radius(b) - 40) <= thresh
    b = s[3]
    assert abs(b[0] - 200) <= thresh
    assert abs(b[1] - 350) <= thresh
    assert abs(radius(b) - 50) <= thresh
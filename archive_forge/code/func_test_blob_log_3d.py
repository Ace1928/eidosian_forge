import math
import numpy as np
import pytest
from numpy.testing import assert_almost_equal
from skimage import feature
from skimage.draw import disk
from skimage.draw.draw3d import ellipsoid
from skimage.feature import blob_dog, blob_doh, blob_log
from skimage.feature.blob import _blob_overlap
def test_blob_log_3d():
    r = 6
    pad = 10
    im3 = ellipsoid(r, r, r)
    im3 = np.pad(im3, pad, mode='constant')
    blobs = blob_log(im3, min_sigma=3, max_sigma=10)
    b = blobs[0]
    assert b.shape == (4,)
    assert b[0] == r + pad + 1
    assert b[1] == r + pad + 1
    assert b[2] == r + pad + 1
    assert abs(math.sqrt(3) * b[3] - r) < 1
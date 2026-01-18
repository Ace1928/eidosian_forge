import math
import numpy as np
import pytest
from numpy.testing import assert_almost_equal
from skimage import feature
from skimage.draw import disk
from skimage.draw.draw3d import ellipsoid
from skimage.feature import blob_dog, blob_doh, blob_log
from skimage.feature.blob import _blob_overlap
def test_blob_log_3d_anisotropic():
    r = 6
    pad = 10
    im3 = ellipsoid(r / 2, r, r)
    im3 = np.pad(im3, pad, mode='constant')
    blobs = blob_log(im3, min_sigma=[1, 2, 2], max_sigma=[5, 10, 10])
    b = blobs[0]
    assert b.shape == (6,)
    assert b[0] == r / 2 + pad + 1
    assert b[1] == r + pad + 1
    assert b[2] == r + pad + 1
    assert abs(math.sqrt(3) * b[3] - r / 2) < 1
    assert abs(math.sqrt(3) * b[4] - r) < 1
    assert abs(math.sqrt(3) * b[5] - r) < 1
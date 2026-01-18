import math
import numpy as np
import pytest
from numpy.testing import assert_almost_equal
from skimage import feature
from skimage.draw import disk
from skimage.draw.draw3d import ellipsoid
from skimage.feature import blob_dog, blob_doh, blob_log
from skimage.feature.blob import _blob_overlap
@pytest.mark.parametrize('dtype', [np.uint8, np.float16, np.float32, np.float64])
@pytest.mark.parametrize('threshold_type', ['absolute', 'relative'])
def test_blob_dog_3d_anisotropic(dtype, threshold_type):
    r = 10
    pad = 10
    im3 = ellipsoid(r / 2, r, r)
    im3 = np.pad(im3, pad, mode='constant')
    if threshold_type == 'absolute':
        threshold = 0.001
        threshold_rel = None
    elif threshold_type == 'relative':
        threshold = None
        threshold_rel = 0.5
    blobs = blob_dog(im3.astype(dtype, copy=False), min_sigma=[1.5, 3, 3], max_sigma=[5, 10, 10], sigma_ratio=1.2, threshold=threshold, threshold_rel=threshold_rel)
    b = blobs[0]
    assert b.shape == (6,)
    assert b[0] == r / 2 + pad + 1
    assert b[1] == r + pad + 1
    assert b[2] == r + pad + 1
    assert abs(math.sqrt(3) * b[3] - r / 2) < 1.1
    assert abs(math.sqrt(3) * b[4] - r) < 1.1
    assert abs(math.sqrt(3) * b[5] - r) < 1.1
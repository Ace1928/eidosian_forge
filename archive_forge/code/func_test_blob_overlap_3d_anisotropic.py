import math
import numpy as np
import pytest
from numpy.testing import assert_almost_equal
from skimage import feature
from skimage.draw import disk
from skimage.draw.draw3d import ellipsoid
from skimage.feature import blob_dog, blob_doh, blob_log
from skimage.feature.blob import _blob_overlap
def test_blob_overlap_3d_anisotropic():
    s3 = math.sqrt(3)
    overlap = _blob_overlap(np.array([0, 0, 0, 2 / s3, 10 / s3, 10 / s3]), np.array([0, 0, 10, 0.2 / s3, 1 / s3, 1 / s3]), sigma_dim=3)
    assert_almost_equal(overlap, 0.48125)
    overlap = _blob_overlap(np.array([0, 0, 0, 2 / s3, 10 / s3, 10 / s3]), np.array([2, 0, 0, 0.2 / s3, 1 / s3, 1 / s3]), sigma_dim=3)
    assert_almost_equal(overlap, 0.48125)
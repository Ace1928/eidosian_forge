import math
import numpy as np
import pytest
from numpy.testing import assert_almost_equal
from skimage import feature
from skimage.draw import disk
from skimage.draw.draw3d import ellipsoid
from skimage.feature import blob_dog, blob_doh, blob_log
from skimage.feature.blob import _blob_overlap
def test_no_blob():
    im = np.zeros((10, 10))
    blobs = blob_log(im, min_sigma=2, max_sigma=5, num_sigma=4)
    assert len(blobs) == 0
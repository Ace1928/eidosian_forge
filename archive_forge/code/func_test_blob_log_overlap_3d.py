import math
import numpy as np
import pytest
from numpy.testing import assert_almost_equal
from skimage import feature
from skimage.draw import disk
from skimage.draw.draw3d import ellipsoid
from skimage.feature import blob_dog, blob_doh, blob_log
from skimage.feature.blob import _blob_overlap
def test_blob_log_overlap_3d():
    r1, r2 = (7, 6)
    pad1, pad2 = (11, 12)
    blob1 = ellipsoid(r1, r1, r1)
    blob1 = np.pad(blob1, pad1, mode='constant')
    blob2 = ellipsoid(r2, r2, r2)
    blob2 = np.pad(blob2, [(pad2, pad2), (pad2 - 9, pad2 + 9), (pad2, pad2)], mode='constant')
    im3 = np.logical_or(blob1, blob2)
    blobs = blob_log(im3, min_sigma=2, max_sigma=10, overlap=0.1)
    assert len(blobs) == 1
import math
import numpy as np
import pytest
from numpy.testing import assert_almost_equal
from skimage import feature
from skimage.draw import disk
from skimage.draw.draw3d import ellipsoid
from skimage.feature import blob_dog, blob_doh, blob_log
from skimage.feature.blob import _blob_overlap
def test_blob_doh_no_peaks():
    img_empty = np.zeros((100, 100))
    assert blob_doh(img_empty).size == 0
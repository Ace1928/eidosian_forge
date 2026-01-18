import math
import numpy as np
import pytest
from numpy.testing import assert_almost_equal
from skimage import feature
from skimage.draw import disk
from skimage.draw.draw3d import ellipsoid
from skimage.feature import blob_dog, blob_doh, blob_log
from skimage.feature.blob import _blob_overlap
@pytest.mark.parametrize('anisotropic', [False, True])
@pytest.mark.parametrize('ndim', [1, 2, 3, 4])
@pytest.mark.parametrize('function_name', ['blob_dog', 'blob_log'])
def test_nd_blob_no_peaks_shape(function_name, ndim, anisotropic):
    z = np.zeros((16,) * ndim, dtype=np.float32)
    if anisotropic:
        max_sigma = 8 + np.arange(ndim)
    else:
        max_sigma = 8
    blob_func = getattr(feature, function_name)
    blobs = blob_func(z, max_sigma=max_sigma)
    expected_shape = 2 * z.ndim if anisotropic else z.ndim + 1
    assert blobs.shape == (0, expected_shape)
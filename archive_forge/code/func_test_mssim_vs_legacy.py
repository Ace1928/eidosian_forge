import numpy as np
import pytest
from numpy.testing import assert_equal, assert_almost_equal
from skimage import data
from skimage._shared._warnings import expected_warnings
from skimage._shared.utils import _supported_float_type
from skimage.metrics import structural_similarity
@pytest.mark.parametrize('dtype', [np.uint8, np.int32, np.float16, np.float32, np.float64])
def test_mssim_vs_legacy(dtype):
    mssim_skimage_0pt17 = 0.3674518327910367
    assert cam.dtype == np.uint8
    assert cam_noisy.dtype == np.uint8
    mssim = structural_similarity(cam.astype(dtype), cam_noisy.astype(dtype), data_range=255)
    assert_almost_equal(mssim, mssim_skimage_0pt17)
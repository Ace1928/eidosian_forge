import numpy as np
import pytest
from numpy.testing import assert_equal, assert_almost_equal
from skimage import data
from skimage._shared._warnings import expected_warnings
from skimage.metrics import (
def test_NRMSE_no_int_overflow():
    camf = cam.astype(np.float32)
    cam_noisyf = cam_noisy.astype(np.float32)
    assert_almost_equal(mean_squared_error(cam, cam_noisy), mean_squared_error(camf, cam_noisyf))
    assert_almost_equal(normalized_root_mse(cam, cam_noisy), normalized_root_mse(camf, cam_noisyf))
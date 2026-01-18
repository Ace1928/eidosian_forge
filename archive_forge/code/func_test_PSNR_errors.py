import numpy as np
import pytest
from numpy.testing import assert_equal, assert_almost_equal
from skimage import data
from skimage._shared._warnings import expected_warnings
from skimage.metrics import (
def test_PSNR_errors():
    with pytest.raises(ValueError):
        peak_signal_noise_ratio(cam, cam[:-1, :])
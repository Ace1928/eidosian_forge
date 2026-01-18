import warnings
import numpy as np
import pytest
from numpy.testing import (
from packaging.version import Version
from skimage import data
from skimage import exposure
from skimage import util
from skimage.color import rgb2gray
from skimage.exposure.exposure import intensity_range
from skimage.util.dtype import dtype_range
from skimage._shared._warnings import expected_warnings
from skimage._shared.utils import _supported_float_type
def test_adapthist_alpha():
    """Test an RGBA color image"""
    img = util.img_as_float(data.astronaut())
    alpha = np.ones((img.shape[0], img.shape[1]), dtype=float)
    img = np.dstack((img, alpha))
    adapted = exposure.equalize_adapthist(img)
    assert adapted.shape != img.shape
    img = img[:, :, :3]
    full_scale = exposure.rescale_intensity(img)
    assert img.shape == adapted.shape
    assert_almost_equal(peak_snr(full_scale, adapted), 109.393, 2)
    assert_almost_equal(norm_brightness_err(full_scale, adapted), 0.0248, 3)
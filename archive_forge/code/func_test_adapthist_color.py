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
def test_adapthist_color():
    """Test an RGB color uint16 image"""
    img = util.img_as_uint(data.astronaut())
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        hist, bin_centers = exposure.histogram(img)
        assert len(w) > 0
    adapted = exposure.equalize_adapthist(img, clip_limit=0.01)
    assert adapted.min() == 0
    assert adapted.max() == 1.0
    assert img.shape == adapted.shape
    full_scale = exposure.rescale_intensity(img)
    assert_almost_equal(peak_snr(full_scale, adapted), 109.393, 1)
    assert_almost_equal(norm_brightness_err(full_scale, adapted), 0.02, 2)
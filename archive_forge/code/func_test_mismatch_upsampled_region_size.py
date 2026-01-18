import itertools
import warnings
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.ndimage import fourier_shift
import scipy.fft as fft
from skimage import img_as_float
from skimage._shared._warnings import expected_warnings
from skimage._shared.testing import assert_stacklevel
from skimage._shared.utils import _supported_float_type
from skimage.data import camera, binary_blobs, eagle
from skimage.registration._phase_cross_correlation import (
def test_mismatch_upsampled_region_size():
    with pytest.raises(ValueError):
        _upsampled_dft(np.ones((4, 4)), upsampled_region_size=[3, 2, 1, 4])
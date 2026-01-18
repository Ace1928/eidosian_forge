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
@pytest.mark.parametrize('disambiguate', [True, False])
def test_disambiguate_zero_shift(disambiguate):
    """When the shift is 0, disambiguation becomes degenerate.

    Some quadrants become size 0, which prevents computation of
    cross-correlation. This test ensures that nothing bad happens in that
    scenario.
    """
    image = camera()
    computed_shift, _, _ = phase_cross_correlation(image, image, disambiguate=disambiguate)
    assert isinstance(computed_shift, np.ndarray)
    np.testing.assert_array_equal(computed_shift, np.array((0.0, 0.0)))
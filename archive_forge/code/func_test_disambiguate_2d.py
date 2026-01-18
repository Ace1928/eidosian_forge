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
@pytest.mark.parametrize(('shift0', 'shift1'), itertools.product((100, -100, 350, -350), (100, -100, 350, -350)))
def test_disambiguate_2d(shift0, shift1):
    image = eagle()[500:, 900:]
    shift = (shift0, shift1)
    origin0 = []
    for s in shift:
        if s > 0:
            origin0.append(0)
        else:
            origin0.append(-s)
    origin1 = np.array(origin0) + shift
    slice0 = tuple((slice(o, o + 450) for o in origin0))
    slice1 = tuple((slice(o, o + 450) for o in origin1))
    reference = image[slice0]
    moving = image[slice1]
    computed_shift, _, _ = phase_cross_correlation(reference, moving, disambiguate=True)
    np.testing.assert_equal(shift, computed_shift)
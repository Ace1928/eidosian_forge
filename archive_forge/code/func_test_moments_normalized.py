import itertools
import numpy as np
import pytest
from scipy import ndimage as ndi
from skimage import draw
from skimage._shared import testing
from skimage._shared.testing import assert_allclose, assert_almost_equal, assert_equal
from skimage._shared.utils import _supported_float_type
from skimage.measure import (
def test_moments_normalized():
    image = np.zeros((20, 20), dtype=np.float64)
    image[13:17, 13:17] = 1
    mu = moments_central(image, (14.5, 14.5))
    nu = moments_normalized(mu)
    image2 = np.zeros((20, 20), dtype=np.float64)
    image2[11:13, 11:13] = 0.7
    mu2 = moments_central(image2, (11.5, 11.5))
    nu2 = moments_normalized(mu2)
    assert_almost_equal(nu, nu2, decimal=1)
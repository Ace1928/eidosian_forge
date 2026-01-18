import itertools
import numpy as np
import pytest
from scipy import ndimage as ndi
from skimage import draw
from skimage._shared import testing
from skimage._shared.testing import assert_allclose, assert_almost_equal, assert_equal
from skimage._shared.utils import _supported_float_type
from skimage.measure import (
def test_moments_hu():
    image = np.zeros((20, 20), dtype=np.float64)
    image[13:15, 13:17] = 1
    mu = moments_central(image, (13.5, 14.5))
    nu = moments_normalized(mu)
    hu = moments_hu(nu)
    image2 = np.zeros((20, 20), dtype=np.float64)
    image2[11, 11:13] = 1
    image2 = image2.T
    mu2 = moments_central(image2, (11.5, 11))
    nu2 = moments_normalized(mu2)
    hu2 = moments_hu(nu2)
    assert_almost_equal(hu, hu2, decimal=1)
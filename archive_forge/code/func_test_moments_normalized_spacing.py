import itertools
import numpy as np
import pytest
from scipy import ndimage as ndi
from skimage import draw
from skimage._shared import testing
from skimage._shared.testing import assert_allclose, assert_almost_equal, assert_equal
from skimage._shared.utils import _supported_float_type
from skimage.measure import (
@pytest.mark.parametrize('anisotropic', [False, True])
def test_moments_normalized_spacing(anisotropic):
    image = np.zeros((20, 20), dtype=np.double)
    image[13:17, 13:17] = 1
    if not anisotropic:
        spacing1 = (1, 1)
        spacing2 = (3, 3)
    else:
        spacing1 = (1, 2)
        spacing2 = (2, 4)
    mu = moments_central(image, spacing=spacing1)
    nu = moments_normalized(mu, spacing=spacing1)
    mu2 = moments_central(image, spacing=spacing2)
    nu2 = moments_normalized(mu2, spacing=spacing2)
    compare_moments(nu, nu2)
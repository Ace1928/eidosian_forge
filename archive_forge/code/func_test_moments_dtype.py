import itertools
import numpy as np
import pytest
from scipy import ndimage as ndi
from skimage import draw
from skimage._shared import testing
from skimage._shared.testing import assert_allclose, assert_almost_equal, assert_equal
from skimage._shared.utils import _supported_float_type
from skimage.measure import (
@pytest.mark.parametrize('dtype', [np.float16, np.float32, np.float64])
def test_moments_dtype(dtype):
    image = np.zeros((20, 20), dtype=dtype)
    image[13:15, 13:17] = 1
    expected_dtype = _supported_float_type(dtype)
    mu = moments_central(image, (13.5, 14.5))
    assert mu.dtype == expected_dtype
    nu = moments_normalized(mu)
    assert nu.dtype == expected_dtype
    hu = moments_hu(nu)
    assert hu.dtype == expected_dtype
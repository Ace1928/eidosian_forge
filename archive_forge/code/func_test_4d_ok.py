import numpy as np
import pytest
from numpy.testing import assert_array_equal
from skimage._shared.utils import _supported_float_type
from skimage._shared.testing import assert_stacklevel
from skimage.filters import difference_of_gaussians, gaussian
def test_4d_ok():
    img = np.zeros((5,) * 4)
    img[2, 2, 2, 2] = 1
    res = gaussian(img, sigma=1, mode='reflect', preserve_range=True)
    assert np.allclose(res.sum(), 1)
import numpy as np
import pytest
from numpy import sqrt, ceil
from numpy.testing import assert_almost_equal
from skimage import data
from skimage import img_as_float
from skimage.feature import daisy
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_daisy_sigmas_and_radii(dtype):
    img = data.astronaut()[:64, :64].mean(axis=2).astype(dtype, copy=False)
    sigmas = [1, 2, 3]
    radii = [1, 2]
    descs = daisy(img, sigmas=sigmas, ring_radii=radii)
    assert descs.dtype == img.dtype
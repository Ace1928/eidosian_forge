import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import ndimage
from skimage.filters import median, rank
from skimage._shared.testing import assert_stacklevel
@pytest.mark.parametrize('dtype', [np.uint8, np.uint16, np.float32, np.float64])
def test_median_preserve_dtype(image, dtype):
    median_image = median(image.astype(dtype), behavior='ndimage')
    assert median_image.dtype == dtype
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import ndimage
from skimage.filters import median, rank
from skimage._shared.testing import assert_stacklevel
def test_median_error_ndim():
    img = np.random.randint(0, 10, size=(5, 5, 5, 5), dtype=np.uint8)
    with pytest.raises(ValueError):
        median(img, behavior='rank')
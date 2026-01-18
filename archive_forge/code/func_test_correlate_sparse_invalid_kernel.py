import numpy as np
import pytest
from numpy.testing import assert_equal
from scipy import ndimage as ndi
from skimage._shared.utils import _supported_float_type
from skimage.filters import correlate_sparse
@pytest.mark.parametrize('mode', ['nearest', 'reflect', 'mirror'])
def test_correlate_sparse_invalid_kernel(mode):
    image = np.array([[0, 0, 1, 3, 5], [0, 1, 4, 3, 4], [1, 2, 5, 4, 1], [2, 4, 5, 2, 1], [4, 5, 1, 0, 0]], dtype=float)
    invalid_kernel = np.array([0, 1, 2, 4]).reshape((2, 2))
    with pytest.raises(ValueError):
        correlate_sparse(image, invalid_kernel, mode=mode)
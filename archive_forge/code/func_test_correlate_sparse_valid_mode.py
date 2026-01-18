import numpy as np
import pytest
from numpy.testing import assert_equal
from scipy import ndimage as ndi
from skimage._shared.utils import _supported_float_type
from skimage.filters import correlate_sparse
def test_correlate_sparse_valid_mode():
    image = np.array([[0, 0, 1, 3, 5], [0, 1, 4, 3, 4], [1, 2, 5, 4, 1], [2, 4, 5, 2, 1], [4, 5, 1, 0, 0]], dtype=float)
    kernel = np.array([0, 1, 2, 4, 8, 16, 32, 64, 128]).reshape((3, 3))
    cs_output = correlate_sparse(image, kernel, mode='valid')
    ndi_output = ndi.correlate(image, kernel, mode='wrap')
    ndi_output = ndi_output[1:4, 1:4]
    assert_equal(cs_output, ndi_output)
import numpy as np
import pytest
from numpy.testing import assert_equal
from scipy import ndimage as ndi
from skimage._shared.utils import _supported_float_type
from skimage.filters import correlate_sparse
@pytest.mark.parametrize('mode', ['nearest', 'reflect', 'mirror'])
@pytest.mark.parametrize('dtype', [np.uint16, np.int32, np.float16, np.float32, np.float64])
def test_correlate_sparse(mode, dtype):
    image = np.array([[0, 0, 1, 3, 5], [0, 1, 4, 3, 4], [1, 2, 5, 4, 1], [2, 4, 5, 2, 1], [4, 5, 1, 0, 0]], dtype=dtype)
    kernel = np.array([0, 1, 2, 4, 8, 16, 32, 64, 128]).reshape((3, 3))
    cs_output = correlate_sparse(image, kernel, mode=mode)
    assert cs_output.dtype == _supported_float_type(image.dtype)
    ndi_output = ndi.correlate(image.astype(float, copy=False), kernel, mode=mode)
    assert_equal(cs_output, ndi_output)
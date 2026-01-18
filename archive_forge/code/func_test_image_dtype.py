import numpy as np
import pytest
from numpy.testing import assert_array_equal
from skimage._shared.utils import _supported_float_type
from skimage._shared.testing import assert_stacklevel
from skimage.filters import difference_of_gaussians, gaussian
@pytest.mark.parametrize('dtype', [np.uint8, np.int32, np.float16, np.float32, np.float64])
def test_image_dtype(dtype):
    a = np.zeros((3, 3), dtype=dtype)
    assert gaussian(a).dtype == _supported_float_type(a.dtype)
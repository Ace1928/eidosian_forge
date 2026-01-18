import numpy as np
from skimage import dtype_limits
from skimage.util.dtype import dtype_range
from skimage.util import invert
from skimage._shared.testing import assert_array_equal
def test_invert_float64_unsigned():
    dtype = 'float64'
    image = np.zeros((3, 3), dtype=dtype)
    lower_dtype_limit, upper_dtype_limit = dtype_limits(image, clip_negative=True)
    image[2, :] = upper_dtype_limit
    expected = np.zeros((3, 3), dtype=dtype)
    expected[0, :] = upper_dtype_limit
    expected[1, :] = upper_dtype_limit
    result = invert(image)
    assert_array_equal(expected, result)
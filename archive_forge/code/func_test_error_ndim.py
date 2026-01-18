from skimage._shared import testing
from skimage._shared.testing import assert_equal, assert_array_equal
import numpy as np
from skimage.util import montage
def test_error_ndim():
    arr_error = np.random.randn(1, 2)
    with testing.raises(ValueError):
        montage(arr_error)
    arr_error = np.random.randn(1, 2, 3, 4)
    with testing.raises(ValueError):
        montage(arr_error)
    arr_error = np.random.randn(1, 2, 3)
    with testing.raises(ValueError):
        montage(arr_error, channel_axis=-1)
    arr_error = np.random.randn(1, 2, 3, 4, 5)
    with testing.raises(ValueError):
        montage(arr_error, channel_axis=-1)
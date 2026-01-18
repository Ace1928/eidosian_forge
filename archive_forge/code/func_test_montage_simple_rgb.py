from skimage._shared import testing
from skimage._shared.testing import assert_equal, assert_array_equal
import numpy as np
from skimage.util import montage
def test_montage_simple_rgb():
    n_images, n_rows, n_cols, n_channels = (2, 2, 2, 2)
    arr_in = np.arange(n_images * n_rows * n_cols * n_channels, dtype=float)
    arr_in = arr_in.reshape(n_images, n_rows, n_cols, n_channels)
    arr_out = montage(arr_in, channel_axis=-1)
    arr_ref = np.array([[[0, 1], [2, 3], [8, 9], [10, 11]], [[4, 5], [6, 7], [12, 13], [14, 15]], [[7, 8], [7, 8], [7, 8], [7, 8]], [[7, 8], [7, 8], [7, 8], [7, 8]]])
    assert_array_equal(arr_out, arr_ref)
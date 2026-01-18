from skimage._shared import testing
from skimage._shared.testing import assert_equal, assert_array_equal
import numpy as np
from skimage.util import montage
def test_montage_grid_custom_gray():
    n_images, n_rows, n_cols = (6, 2, 2)
    arr_in = np.arange(n_images * n_rows * n_cols, dtype=np.float32)
    arr_in = arr_in.reshape(n_images, n_rows, n_cols)
    arr_out = montage(arr_in, grid_shape=(3, 2))
    arr_ref = np.array([[0.0, 1.0, 4.0, 5.0], [2.0, 3.0, 6.0, 7.0], [8.0, 9.0, 12.0, 13.0], [10.0, 11.0, 14.0, 15.0], [16.0, 17.0, 20.0, 21.0], [18.0, 19.0, 22.0, 23.0]])
    assert_array_equal(arr_out, arr_ref)
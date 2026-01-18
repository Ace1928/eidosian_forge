from skimage._shared import testing
from skimage._shared.testing import assert_equal, assert_array_equal
import numpy as np
from skimage.util import montage
def test_montage_grid_default_gray():
    n_images, n_rows, n_cols = (15, 11, 7)
    arr_in = np.arange(n_images * n_rows * n_cols, dtype=float)
    arr_in = arr_in.reshape(n_images, n_rows, n_cols)
    n_tiles = int(np.ceil(np.sqrt(n_images)))
    arr_out = montage(arr_in)
    assert_equal(arr_out.shape, (n_tiles * n_rows, n_tiles * n_cols))
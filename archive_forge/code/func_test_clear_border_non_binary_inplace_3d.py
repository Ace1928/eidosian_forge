import numpy as np
from skimage.segmentation import clear_border
from skimage._shared.testing import assert_array_equal, assert_
def test_clear_border_non_binary_inplace_3d():
    image3d = np.array([[[1, 2, 3, 1, 2], [3, 3, 3, 4, 2], [3, 4, 3, 4, 2], [3, 3, 2, 1, 2]], [[1, 2, 3, 1, 2], [3, 3, 5, 4, 2], [3, 4, 5, 4, 2], [3, 3, 2, 1, 2]], [[1, 2, 3, 1, 2], [3, 3, 3, 4, 2], [3, 4, 3, 4, 2], [3, 3, 2, 1, 2]]])
    result = clear_border(image3d, out=image3d)
    expected = np.array([[[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0], [0, 0, 5, 0, 0], [0, 0, 5, 0, 0], [0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]])
    assert_array_equal(result, expected)
    assert_array_equal(image3d, result)
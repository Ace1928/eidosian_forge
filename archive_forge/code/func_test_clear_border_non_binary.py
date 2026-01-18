import numpy as np
from skimage.segmentation import clear_border
from skimage._shared.testing import assert_array_equal, assert_
def test_clear_border_non_binary():
    image = np.array([[1, 2, 3, 1, 2], [3, 3, 5, 4, 2], [3, 4, 5, 4, 2], [3, 3, 2, 1, 2]])
    result = clear_border(image)
    expected = np.array([[0, 0, 0, 0, 0], [0, 0, 5, 4, 0], [0, 4, 5, 4, 0], [0, 0, 0, 0, 0]])
    assert_array_equal(result, expected)
    assert_(not np.all(image == result))
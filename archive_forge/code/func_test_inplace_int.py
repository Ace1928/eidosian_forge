import numpy as np
import pytest
from skimage.morphology import flood, flood_fill
def test_inplace_int():
    image = np.array([[0, 0, 0, 0, 0, 0, 0], [0, 1, 1, 0, 2, 2, 0], [0, 1, 1, 0, 2, 2, 0], [1, 0, 0, 0, 0, 0, 3], [0, 1, 1, 1, 3, 3, 4]])
    flood_fill(image, (0, 0), 5, in_place=True)
    expected = np.array([[5, 5, 5, 5, 5, 5, 5], [5, 1, 1, 5, 2, 2, 5], [5, 1, 1, 5, 2, 2, 5], [1, 5, 5, 5, 5, 5, 3], [5, 1, 1, 1, 3, 3, 4]])
    np.testing.assert_array_equal(image, expected)
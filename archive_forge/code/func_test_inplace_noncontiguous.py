import numpy as np
import pytest
from skimage.morphology import flood, flood_fill
def test_inplace_noncontiguous():
    image = np.array([[0, 0, 0, 0, 0, 0, 0], [0, 1, 1, 0, 2, 2, 0], [0, 1, 1, 0, 2, 2, 0], [1, 0, 0, 0, 0, 0, 3], [0, 1, 1, 1, 3, 3, 4]])
    image2 = image[::2, ::2]
    flood_fill(image2, (0, 0), 5, in_place=True)
    expected2 = np.array([[5, 5, 5, 5], [5, 1, 2, 5], [5, 1, 3, 4]])
    np.testing.assert_allclose(image2, expected2)
    expected = np.array([[5, 0, 5, 0, 5, 0, 5], [0, 1, 1, 0, 2, 2, 0], [5, 1, 1, 0, 2, 2, 5], [1, 0, 0, 0, 0, 0, 3], [5, 1, 1, 1, 3, 3, 4]])
    np.testing.assert_allclose(image, expected)
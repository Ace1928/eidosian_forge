import numpy as np
import pytest
from skimage.morphology import flood, flood_fill
def test_non_adjacent_footprint():
    footprint = np.array([[1, 0, 0, 0, 1], [0, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0], [1, 0, 0, 0, 1]])
    output = flood_fill(np.zeros((5, 6), dtype=np.uint8), (2, 3), 255, footprint=footprint)
    expected = np.array([[0, 255, 0, 0, 0, 255], [0, 0, 0, 0, 0, 0], [0, 0, 0, 255, 0, 0], [0, 0, 0, 0, 0, 0], [0, 255, 0, 0, 0, 255]], dtype=np.uint8)
    np.testing.assert_equal(output, expected)
    footprint = np.array([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]])
    image = np.zeros((5, 10), dtype=np.uint8)
    image[:, (3, 7, 8)] = 100
    output = flood_fill(image, (0, 0), 255, footprint=footprint)
    expected = np.array([[255, 255, 255, 100, 255, 255, 255, 100, 100, 0], [255, 255, 255, 100, 255, 255, 255, 100, 100, 0], [255, 255, 255, 100, 255, 255, 255, 100, 100, 0], [255, 255, 255, 100, 255, 255, 255, 100, 100, 0], [255, 255, 255, 100, 255, 255, 255, 100, 100, 0]], dtype=np.uint8)
    np.testing.assert_equal(output, expected)
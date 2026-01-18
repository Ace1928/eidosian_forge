import numpy as np
import pytest
from skimage.morphology import flood, flood_fill
def test_footprint():
    footprint = np.array([[0, 1, 1], [0, 1, 1], [0, 0, 0]])
    output = flood_fill(np.zeros((5, 6), dtype=np.uint8), (3, 1), 255, footprint=footprint)
    expected = np.array([[0, 255, 255, 255, 255, 255], [0, 255, 255, 255, 255, 255], [0, 255, 255, 255, 255, 255], [0, 255, 255, 255, 255, 255], [0, 0, 0, 0, 0, 0]], dtype=np.uint8)
    np.testing.assert_equal(output, expected)
    footprint = np.array([[0, 0, 0], [1, 1, 0], [1, 1, 0]])
    output = flood_fill(np.zeros((5, 6), dtype=np.uint8), (1, 4), 255, footprint=footprint)
    expected = np.array([[0, 0, 0, 0, 0, 0], [255, 255, 255, 255, 255, 0], [255, 255, 255, 255, 255, 0], [255, 255, 255, 255, 255, 0], [255, 255, 255, 255, 255, 0]], dtype=np.uint8)
    np.testing.assert_equal(output, expected)
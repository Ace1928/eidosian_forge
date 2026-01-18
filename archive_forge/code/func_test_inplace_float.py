import numpy as np
import pytest
from skimage.morphology import flood, flood_fill
def test_inplace_float():
    image = np.array([[0, 0, 0, 0, 0, 0, 0], [0, 1, 1, 0, 2, 2, 0], [0, 1, 1, 0, 2, 2, 0], [1, 0, 0, 0, 0, 0, 3], [0, 1, 1, 1, 3, 3, 4]], dtype=np.float32)
    flood_fill(image, (0, 0), 5, in_place=True)
    expected = np.array([[5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0], [5.0, 1.0, 1.0, 5.0, 2.0, 2.0, 5.0], [5.0, 1.0, 1.0, 5.0, 2.0, 2.0, 5.0], [1.0, 5.0, 5.0, 5.0, 5.0, 5.0, 3.0], [5.0, 1.0, 1.0, 1.0, 3.0, 3.0, 4.0]], dtype=np.float32)
    np.testing.assert_allclose(image, expected)
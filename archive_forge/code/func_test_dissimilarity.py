import numpy as np
import pytest
from skimage._shared.testing import expected_warnings, run_in_parallel
from skimage.feature import (
from skimage.transform import integral_image
def test_dissimilarity(self):
    result = graycomatrix(self.image, [1], [0, np.pi / 2], 4, normed=True, symmetric=True)
    result = np.round(result, 3)
    dissimilarity = graycoprops(result, 'dissimilarity')
    np.testing.assert_almost_equal(dissimilarity[0, 0], 0.418, decimal=3)
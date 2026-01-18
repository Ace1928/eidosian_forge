import numpy as np
import pytest
from skimage._shared.testing import expected_warnings, run_in_parallel
from skimage.feature import (
from skimage.transform import integral_image
def test_contrast(self):
    result = graycomatrix(self.image, [1, 2], [0], 4, normed=True, symmetric=True)
    result = np.round(result, 3)
    contrast = graycoprops(result, 'contrast')
    np.testing.assert_almost_equal(contrast[0, 0], 0.585, decimal=3)
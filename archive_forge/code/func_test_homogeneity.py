import numpy as np
import pytest
from skimage._shared.testing import expected_warnings, run_in_parallel
from skimage.feature import (
from skimage.transform import integral_image
def test_homogeneity(self):
    result = graycomatrix(self.image, [1], [0, 6], 4, normed=True, symmetric=True)
    homogeneity = graycoprops(result, 'homogeneity')[0, 0]
    np.testing.assert_almost_equal(homogeneity, 0.80833333)
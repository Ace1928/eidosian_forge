from numpy.testing import (assert_allclose, assert_almost_equal,
import numpy as np
import pytest
from matplotlib import mlab
def test_evaluate_diff_dim(self):
    """
        Test the evaluate method when the dim's of dataset and points have
        different dimensions.
        """
    x1 = np.arange(3, 10, 2)
    kde = mlab.GaussianKDE(x1)
    x2 = np.arange(3, 12, 2)
    y_expected = [0.08797252, 0.11774109, 0.11774109, 0.08797252, 0.0370153]
    y = kde.evaluate(x2)
    np.testing.assert_array_almost_equal(y, y_expected, 7)
import numpy as np
import pytest
from skimage._shared import testing
from skimage._shared._warnings import expected_warnings
from skimage._shared.testing import (
from skimage.measure import CircleModel, EllipseModel, LineModelND, ransac
from skimage.measure.fit import _dynamic_max_trials
from skimage.transform import AffineTransform
def test_circle_model_residuals():
    model = CircleModel()
    model.params = (0, 0, 5)
    assert_almost_equal(abs(model.residuals(np.array([[5, 0]]))), 0)
    assert_almost_equal(abs(model.residuals(np.array([[6, 6]]))), np.sqrt(2 * 6 ** 2) - 5)
    assert_almost_equal(abs(model.residuals(np.array([[10, 0]]))), 5)
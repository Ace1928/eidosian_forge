import numpy as np
import pytest
from skimage._shared import testing
from skimage._shared._warnings import expected_warnings
from skimage._shared.testing import (
from skimage.measure import CircleModel, EllipseModel, LineModelND, ransac
from skimage.measure.fit import _dynamic_max_trials
from skimage.transform import AffineTransform
def test_ellipse_model_estimate_from_far_shifted_data():
    params = np.array([1000000.0, 2000000.0, 0.5, 0.1, 0.5], dtype=np.float64)
    angles = np.array([0.107, 0.407, 1.108, 1.489, 2.216, 2.768, 3.183, 3.969, 4.84, 5.387, 5.792, 6.139], dtype=np.float64)
    data = EllipseModel().predict_xy(angles, params=params)
    model = EllipseModel()
    assert model.estimate(data.astype(np.float64))
    assert_almost_equal(params, model.params)
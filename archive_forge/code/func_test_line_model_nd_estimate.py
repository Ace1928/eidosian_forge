import numpy as np
import pytest
from skimage._shared import testing
from skimage._shared._warnings import expected_warnings
from skimage._shared.testing import (
from skimage.measure import CircleModel, EllipseModel, LineModelND, ransac
from skimage.measure.fit import _dynamic_max_trials
from skimage.transform import AffineTransform
def test_line_model_nd_estimate():
    model0 = LineModelND()
    model0.params = (np.array([0, 0, 0], dtype='float'), np.array([1, 1, 1], dtype='float') / np.sqrt(3))
    data0 = model0.params[0] + 10 * np.arange(-100, 100)[..., np.newaxis] * model0.params[1]
    rng = np.random.default_rng(1234)
    data = data0 + rng.normal(size=data0.shape)
    model_est = LineModelND()
    model_est.estimate(data)
    assert_almost_equal(np.linalg.norm(np.cross(model0.params[1], model_est.params[1])), 0, 1)
    a = model_est.params[0] - model0.params[0]
    if np.linalg.norm(a) > 0:
        a /= np.linalg.norm(a)
    assert_almost_equal(np.linalg.norm(np.cross(model0.params[1], a)), 0, 1)
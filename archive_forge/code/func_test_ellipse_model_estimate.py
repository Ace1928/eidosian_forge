import numpy as np
import pytest
from skimage._shared import testing
from skimage._shared._warnings import expected_warnings
from skimage._shared.testing import (
from skimage.measure import CircleModel, EllipseModel, LineModelND, ransac
from skimage.measure.fit import _dynamic_max_trials
from skimage.transform import AffineTransform
def test_ellipse_model_estimate():
    for angle in range(0, 180, 15):
        rad = np.deg2rad(angle)
        model0 = EllipseModel()
        model0.params = (10, 20, 15, 25, rad)
        t = np.linspace(0, 2 * np.pi, 100)
        data0 = model0.predict_xy(t)
        rng = np.random.default_rng(1234)
        data = data0 + rng.normal(size=data0.shape)
        model_est = EllipseModel()
        model_est.estimate(data)
        assert_almost_equal(model0.params[:2], model_est.params[:2], 0)
        res = model_est.residuals(data0)
        assert_array_less(res, np.ones(res.shape))
import numpy as np
import pytest
from skimage._shared import testing
from skimage._shared._warnings import expected_warnings
from skimage._shared.testing import (
from skimage.measure import CircleModel, EllipseModel, LineModelND, ransac
from skimage.measure.fit import _dynamic_max_trials
from skimage.transform import AffineTransform
def test_ellipse_parameter_stability():
    """The fit should be modified so that a > b"""
    for angle in np.arange(0, 180 + 1, 1):
        theta = np.deg2rad(angle)
        c = np.cos(theta)
        s = np.sin(theta)
        R = np.array([[c, -s], [s, c]])
        t = np.linspace(0, 2 * np.pi, 20)
        a = 100
        b = 50
        points = np.array([a * np.cos(t), b * np.sin(t)])
        points = R @ points
        ellipse_model = EllipseModel()
        ellipse_model.estimate(points.T)
        _, _, a_prime, b_prime, theta_prime = ellipse_model.params
        assert_almost_equal(theta_prime, theta)
        assert_almost_equal(a_prime, a)
        assert_almost_equal(b_prime, b)
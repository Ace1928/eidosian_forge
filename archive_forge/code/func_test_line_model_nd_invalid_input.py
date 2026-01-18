import numpy as np
import pytest
from skimage._shared import testing
from skimage._shared._warnings import expected_warnings
from skimage._shared.testing import (
from skimage.measure import CircleModel, EllipseModel, LineModelND, ransac
from skimage.measure.fit import _dynamic_max_trials
from skimage.transform import AffineTransform
def test_line_model_nd_invalid_input():
    with testing.raises(ValueError):
        LineModelND().predict_x(np.zeros(1))
    with testing.raises(ValueError):
        LineModelND().predict_y(np.zeros(1))
    with testing.raises(ValueError):
        LineModelND().predict_x(np.zeros(1), np.zeros(1))
    with testing.raises(ValueError):
        LineModelND().predict_y(np.zeros(1))
    with testing.raises(ValueError):
        LineModelND().predict_y(np.zeros(1), np.zeros(1))
    assert not LineModelND().estimate(np.empty((1, 3)))
    assert not LineModelND().estimate(np.empty((1, 2)))
    with testing.raises(ValueError):
        LineModelND().residuals(np.empty((1, 3)))
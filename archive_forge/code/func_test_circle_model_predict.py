import numpy as np
import pytest
from skimage._shared import testing
from skimage._shared._warnings import expected_warnings
from skimage._shared.testing import (
from skimage.measure import CircleModel, EllipseModel, LineModelND, ransac
from skimage.measure.fit import _dynamic_max_trials
from skimage.transform import AffineTransform
def test_circle_model_predict():
    model = CircleModel()
    r = 5
    model.params = (0, 0, r)
    t = np.arange(0, 2 * np.pi, np.pi / 2)
    xy = np.array(((5, 0), (0, 5), (-5, 0), (0, -5)))
    assert_almost_equal(xy, model.predict_xy(t))
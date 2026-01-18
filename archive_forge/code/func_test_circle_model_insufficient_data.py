import numpy as np
import pytest
from skimage._shared import testing
from skimage._shared._warnings import expected_warnings
from skimage._shared.testing import (
from skimage.measure import CircleModel, EllipseModel, LineModelND, ransac
from skimage.measure.fit import _dynamic_max_trials
from skimage.transform import AffineTransform
def test_circle_model_insufficient_data():
    model = CircleModel()
    warning_message = ['Input does not contain enough significant data points.']
    with expected_warnings(warning_message):
        model.estimate(np.array([[1, 2], [3, 4]]))
    with expected_warnings(warning_message):
        model.estimate(np.array([[0, 0], [1, 1], [2, 2]]))
    warning_message = 'Standard deviation of data is too small to estimate circle with meaningful precision.'
    with pytest.warns(RuntimeWarning, match=warning_message) as _warnings:
        assert not model.estimate(np.ones((6, 2)))
    assert_stacklevel(_warnings)
    assert len(_warnings) == 1
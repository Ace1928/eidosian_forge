import numpy as np
import pytest
from sklearn.datasets import load_iris
from sklearn.model_selection import (
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import shuffle
from sklearn.utils._testing import assert_allclose, assert_array_equal
def test_learning_curve_display_deprecate_log_scale(data, pyplot):
    """Check that we warn for the deprecated parameter `log_scale`."""
    X, y = data
    estimator = DecisionTreeClassifier(random_state=0)
    with pytest.warns(FutureWarning, match='`log_scale` parameter is deprecated'):
        display = LearningCurveDisplay.from_estimator(estimator, X, y, train_sizes=[0.3, 0.6, 0.9], log_scale=True)
    assert display.ax_.get_xscale() == 'log'
    assert display.ax_.get_yscale() == 'linear'
    with pytest.warns(FutureWarning, match='`log_scale` parameter is deprecated'):
        display = LearningCurveDisplay.from_estimator(estimator, X, y, train_sizes=[0.3, 0.6, 0.9], log_scale=False)
    assert display.ax_.get_xscale() == 'linear'
    assert display.ax_.get_yscale() == 'linear'
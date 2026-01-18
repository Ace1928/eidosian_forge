import numpy as np
import pytest
from sklearn.datasets import load_iris
from sklearn.model_selection import (
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import shuffle
from sklearn.utils._testing import assert_allclose, assert_array_equal
@pytest.mark.parametrize('score_name, ylabel', [(None, 'Score'), ('Accuracy', 'Accuracy')])
@pytest.mark.parametrize('CurveDisplay, specific_params', [(ValidationCurveDisplay, {'param_name': 'max_depth', 'param_range': [1, 3, 5]}), (LearningCurveDisplay, {'train_sizes': [0.3, 0.6, 0.9]})])
def test_curve_display_score_name(pyplot, data, score_name, ylabel, CurveDisplay, specific_params):
    """Check that we can overwrite the default score name shown on the y-axis."""
    X, y = data
    estimator = DecisionTreeClassifier(random_state=0)
    display = CurveDisplay.from_estimator(estimator, X, y, **specific_params, score_name=score_name)
    assert display.ax_.get_ylabel() == ylabel
    X, y = data
    estimator = DecisionTreeClassifier(max_depth=1, random_state=0)
    display = CurveDisplay.from_estimator(estimator, X, y, **specific_params, score_name=score_name)
    assert display.score_name == ylabel
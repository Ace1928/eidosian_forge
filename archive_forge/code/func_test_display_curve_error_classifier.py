import numpy as np
import pytest
from sklearn.base import ClassifierMixin, clone
from sklearn.calibration import CalibrationDisplay
from sklearn.compose import make_column_transformer
from sklearn.datasets import load_iris
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
@pytest.mark.parametrize('Display', [CalibrationDisplay, DetCurveDisplay, PrecisionRecallDisplay, RocCurveDisplay])
def test_display_curve_error_classifier(pyplot, data, data_binary, Display):
    """Check that a proper error is raised when only binary classification is
    supported."""
    X, y = data
    X_binary, y_binary = data_binary
    clf = DecisionTreeClassifier().fit(X, y)
    msg = "Expected 'estimator' to be a binary classifier. Got 3 classes instead."
    with pytest.raises(ValueError, match=msg):
        Display.from_estimator(clf, X, y)
    with pytest.raises(ValueError, match=msg):
        Display.from_estimator(clf, X_binary, y_binary)
    clf = DecisionTreeClassifier().fit(X_binary, y_binary)
    msg = 'The target y is not binary. Got multiclass type of target.'
    with pytest.raises(ValueError, match=msg):
        Display.from_estimator(clf, X, y)
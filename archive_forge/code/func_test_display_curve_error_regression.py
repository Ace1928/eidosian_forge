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
def test_display_curve_error_regression(pyplot, data_binary, Display):
    """Check that we raise an error with regressor."""
    X, y = data_binary
    regressor = DecisionTreeRegressor().fit(X, y)
    msg = "Expected 'estimator' to be a binary classifier. Got DecisionTreeRegressor"
    with pytest.raises(ValueError, match=msg):
        Display.from_estimator(regressor, X, y)
    classifier = DecisionTreeClassifier().fit(X, y)
    y = y + 0.5
    msg = 'The target y is not binary. Got continuous type of target.'
    with pytest.raises(ValueError, match=msg):
        Display.from_estimator(classifier, X, y)
    with pytest.raises(ValueError, match=msg):
        Display.from_predictions(y, regressor.fit(X, y).predict(X))
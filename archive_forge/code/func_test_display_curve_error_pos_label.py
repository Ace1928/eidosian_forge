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
@pytest.mark.parametrize('Display', [DetCurveDisplay, PrecisionRecallDisplay, RocCurveDisplay])
def test_display_curve_error_pos_label(pyplot, data_binary, Display):
    """Check consistence of error message when `pos_label` should be specified."""
    X, y = data_binary
    y = y + 10
    classifier = DecisionTreeClassifier().fit(X, y)
    y_pred = classifier.predict_proba(X)[:, -1]
    msg = 'y_true takes value in {10, 11} and pos_label is not specified'
    with pytest.raises(ValueError, match=msg):
        Display.from_predictions(y, y_pred)
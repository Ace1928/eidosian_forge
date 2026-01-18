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
@pytest.mark.parametrize('Display', [CalibrationDisplay, DetCurveDisplay, PrecisionRecallDisplay, RocCurveDisplay, PredictionErrorDisplay, ConfusionMatrixDisplay])
@pytest.mark.parametrize('constructor', ['from_predictions', 'from_estimator'])
def test_classifier_display_curve_named_constructor_return_type(pyplot, data_binary, Display, constructor):
    """Check that named constructors return the correct type when subclassed.

    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/pull/27675
    """
    X, y = data_binary
    y_pred = y
    classifier = LogisticRegression().fit(X, y)

    class SubclassOfDisplay(Display):
        pass
    if constructor == 'from_predictions':
        curve = SubclassOfDisplay.from_predictions(y, y_pred)
    else:
        curve = SubclassOfDisplay.from_estimator(classifier, X, y)
    assert isinstance(curve, SubclassOfDisplay)
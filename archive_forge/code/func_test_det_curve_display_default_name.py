import numpy as np
import pytest
from numpy.testing import assert_allclose
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import DetCurveDisplay, det_curve
@pytest.mark.parametrize('constructor_name, expected_clf_name', [('from_estimator', 'LogisticRegression'), ('from_predictions', 'Classifier')])
def test_det_curve_display_default_name(pyplot, constructor_name, expected_clf_name):
    X, y = load_iris(return_X_y=True)
    X, y = (X[y < 2], y[y < 2])
    lr = LogisticRegression().fit(X, y)
    y_pred = lr.predict_proba(X)[:, 1]
    if constructor_name == 'from_estimator':
        disp = DetCurveDisplay.from_estimator(lr, X, y)
    else:
        disp = DetCurveDisplay.from_predictions(y, y_pred)
    assert disp.estimator_name == expected_clf_name
    assert disp.line_.get_label() == expected_clf_name
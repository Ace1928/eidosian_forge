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
@pytest.mark.parametrize('constructor_name', ['from_estimator', 'from_predictions'])
def test_display_curve_estimator_name_multiple_calls(pyplot, data_binary, Display, constructor_name):
    """Check that passing `name` when calling `plot` will overwrite the original name
    in the legend."""
    X, y = data_binary
    clf_name = 'my hand-crafted name'
    clf = LogisticRegression().fit(X, y)
    y_pred = clf.predict_proba(X)[:, 1]
    assert constructor_name in ('from_estimator', 'from_predictions')
    if constructor_name == 'from_estimator':
        disp = Display.from_estimator(clf, X, y, name=clf_name)
    else:
        disp = Display.from_predictions(y, y_pred, name=clf_name)
    assert disp.estimator_name == clf_name
    pyplot.close('all')
    disp.plot()
    assert clf_name in disp.line_.get_label()
    pyplot.close('all')
    clf_name = 'another_name'
    disp.plot(name=clf_name)
    assert clf_name in disp.line_.get_label()
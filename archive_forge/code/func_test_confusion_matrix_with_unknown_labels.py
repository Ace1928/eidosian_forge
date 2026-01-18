import numpy as np
import pytest
from numpy.testing import (
from sklearn.compose import make_column_transformer
from sklearn.datasets import make_classification
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR
@pytest.mark.parametrize('constructor_name', ['from_estimator', 'from_predictions'])
def test_confusion_matrix_with_unknown_labels(pyplot, constructor_name):
    """Check that when labels=None, the unique values in `y_pred` and `y_true`
    will be used.
    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/pull/18405
    """
    n_classes = 5
    X, y = make_classification(n_samples=100, n_informative=5, n_classes=n_classes, random_state=0)
    classifier = SVC().fit(X, y)
    y_pred = classifier.predict(X)
    y = y + 1
    assert constructor_name in ('from_estimator', 'from_predictions')
    common_kwargs = {'labels': None}
    if constructor_name == 'from_estimator':
        disp = ConfusionMatrixDisplay.from_estimator(classifier, X, y, **common_kwargs)
    else:
        disp = ConfusionMatrixDisplay.from_predictions(y, y_pred, **common_kwargs)
    display_labels = [tick.get_text() for tick in disp.ax_.get_xticklabels()]
    expected_labels = [str(i) for i in range(n_classes + 1)]
    assert_array_equal(expected_labels, display_labels)
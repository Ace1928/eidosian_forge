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
@pytest.mark.parametrize('with_labels', [True, False])
@pytest.mark.parametrize('with_display_labels', [True, False])
def test_confusion_matrix_display_custom_labels(pyplot, constructor_name, with_labels, with_display_labels):
    """Check the resulting plot when labels are given."""
    n_classes = 5
    X, y = make_classification(n_samples=100, n_informative=5, n_classes=n_classes, random_state=0)
    classifier = SVC().fit(X, y)
    y_pred = classifier.predict(X)
    assert constructor_name in ('from_estimator', 'from_predictions')
    ax = pyplot.gca()
    labels = [2, 1, 0, 3, 4] if with_labels else None
    display_labels = ['b', 'd', 'a', 'e', 'f'] if with_display_labels else None
    cm = confusion_matrix(y, y_pred, labels=labels)
    common_kwargs = {'ax': ax, 'display_labels': display_labels, 'labels': labels}
    if constructor_name == 'from_estimator':
        disp = ConfusionMatrixDisplay.from_estimator(classifier, X, y, **common_kwargs)
    else:
        disp = ConfusionMatrixDisplay.from_predictions(y, y_pred, **common_kwargs)
    assert_allclose(disp.confusion_matrix, cm)
    if with_display_labels:
        expected_display_labels = display_labels
    elif with_labels:
        expected_display_labels = labels
    else:
        expected_display_labels = list(range(n_classes))
    expected_display_labels_str = [str(name) for name in expected_display_labels]
    x_ticks = [tick.get_text() for tick in disp.ax_.get_xticklabels()]
    y_ticks = [tick.get_text() for tick in disp.ax_.get_yticklabels()]
    assert_array_equal(disp.display_labels, expected_display_labels)
    assert_array_equal(x_ticks, expected_display_labels_str)
    assert_array_equal(y_ticks, expected_display_labels_str)
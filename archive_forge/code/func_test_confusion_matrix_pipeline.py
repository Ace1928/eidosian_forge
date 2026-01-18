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
@pytest.mark.parametrize('clf', [LogisticRegression(), make_pipeline(StandardScaler(), LogisticRegression()), make_pipeline(make_column_transformer((StandardScaler(), [0, 1])), LogisticRegression())], ids=['clf', 'pipeline-clf', 'pipeline-column_transformer-clf'])
def test_confusion_matrix_pipeline(pyplot, clf):
    """Check the behaviour of the plotting with more complex pipeline."""
    n_classes = 5
    X, y = make_classification(n_samples=100, n_informative=5, n_classes=n_classes, random_state=0)
    with pytest.raises(NotFittedError):
        ConfusionMatrixDisplay.from_estimator(clf, X, y)
    clf.fit(X, y)
    y_pred = clf.predict(X)
    disp = ConfusionMatrixDisplay.from_estimator(clf, X, y)
    cm = confusion_matrix(y, y_pred)
    assert_allclose(disp.confusion_matrix, cm)
    assert disp.text_.shape == (n_classes, n_classes)
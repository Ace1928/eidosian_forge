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
def test_confusion_matrix_display(pyplot, constructor_name):
    """Check the behaviour of the default constructor without using the class
    methods."""
    n_classes = 5
    X, y = make_classification(n_samples=100, n_informative=5, n_classes=n_classes, random_state=0)
    classifier = SVC().fit(X, y)
    y_pred = classifier.predict(X)
    assert constructor_name in ('from_estimator', 'from_predictions')
    cm = confusion_matrix(y, y_pred)
    common_kwargs = {'normalize': None, 'include_values': True, 'cmap': 'viridis', 'xticks_rotation': 45.0}
    if constructor_name == 'from_estimator':
        disp = ConfusionMatrixDisplay.from_estimator(classifier, X, y, **common_kwargs)
    else:
        disp = ConfusionMatrixDisplay.from_predictions(y, y_pred, **common_kwargs)
    assert_allclose(disp.confusion_matrix, cm)
    assert disp.text_.shape == (n_classes, n_classes)
    rotations = [tick.get_rotation() for tick in disp.ax_.get_xticklabels()]
    assert_allclose(rotations, 45.0)
    image_data = disp.im_.get_array().data
    assert_allclose(image_data, cm)
    disp.plot(cmap='plasma')
    assert disp.im_.get_cmap().name == 'plasma'
    disp.plot(include_values=False)
    assert disp.text_ is None
    disp.plot(xticks_rotation=90.0)
    rotations = [tick.get_rotation() for tick in disp.ax_.get_xticklabels()]
    assert_allclose(rotations, 90.0)
    disp.plot(values_format='e')
    expected_text = np.array([format(v, 'e') for v in cm.ravel(order='C')])
    text_text = np.array([t.get_text() for t in disp.text_.ravel(order='C')])
    assert_array_equal(expected_text, text_text)
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
@pytest.mark.parametrize('normalize', ['true', 'pred', 'all', None])
@pytest.mark.parametrize('include_values', [True, False])
def test_confusion_matrix_display_plotting(pyplot, constructor_name, normalize, include_values):
    """Check the overall plotting rendering."""
    n_classes = 5
    X, y = make_classification(n_samples=100, n_informative=5, n_classes=n_classes, random_state=0)
    classifier = SVC().fit(X, y)
    y_pred = classifier.predict(X)
    assert constructor_name in ('from_estimator', 'from_predictions')
    ax = pyplot.gca()
    cmap = 'plasma'
    cm = confusion_matrix(y, y_pred)
    common_kwargs = {'normalize': normalize, 'cmap': cmap, 'ax': ax, 'include_values': include_values}
    if constructor_name == 'from_estimator':
        disp = ConfusionMatrixDisplay.from_estimator(classifier, X, y, **common_kwargs)
    else:
        disp = ConfusionMatrixDisplay.from_predictions(y, y_pred, **common_kwargs)
    assert disp.ax_ == ax
    if normalize == 'true':
        cm = cm / cm.sum(axis=1, keepdims=True)
    elif normalize == 'pred':
        cm = cm / cm.sum(axis=0, keepdims=True)
    elif normalize == 'all':
        cm = cm / cm.sum()
    assert_allclose(disp.confusion_matrix, cm)
    import matplotlib as mpl
    assert isinstance(disp.im_, mpl.image.AxesImage)
    assert disp.im_.get_cmap().name == cmap
    assert isinstance(disp.ax_, pyplot.Axes)
    assert isinstance(disp.figure_, pyplot.Figure)
    assert disp.ax_.get_ylabel() == 'True label'
    assert disp.ax_.get_xlabel() == 'Predicted label'
    x_ticks = [tick.get_text() for tick in disp.ax_.get_xticklabels()]
    y_ticks = [tick.get_text() for tick in disp.ax_.get_yticklabels()]
    expected_display_labels = list(range(n_classes))
    expected_display_labels_str = [str(name) for name in expected_display_labels]
    assert_array_equal(disp.display_labels, expected_display_labels)
    assert_array_equal(x_ticks, expected_display_labels_str)
    assert_array_equal(y_ticks, expected_display_labels_str)
    image_data = disp.im_.get_array().data
    assert_allclose(image_data, cm)
    if include_values:
        assert disp.text_.shape == (n_classes, n_classes)
        fmt = '.2g'
        expected_text = np.array([format(v, fmt) for v in cm.ravel(order='C')])
        text_text = np.array([t.get_text() for t in disp.text_.ravel(order='C')])
        assert_array_equal(expected_text, text_text)
    else:
        assert disp.text_ is None
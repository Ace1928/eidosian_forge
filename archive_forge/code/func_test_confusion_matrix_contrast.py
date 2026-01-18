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
def test_confusion_matrix_contrast(pyplot):
    """Check that the text color is appropriate depending on background."""
    cm = np.eye(2) / 2
    disp = ConfusionMatrixDisplay(cm, display_labels=[0, 1])
    disp.plot(cmap=pyplot.cm.gray)
    assert_allclose(disp.text_[0, 0].get_color(), [0.0, 0.0, 0.0, 1.0])
    assert_allclose(disp.text_[1, 1].get_color(), [0.0, 0.0, 0.0, 1.0])
    assert_allclose(disp.text_[0, 1].get_color(), [1.0, 1.0, 1.0, 1.0])
    assert_allclose(disp.text_[1, 0].get_color(), [1.0, 1.0, 1.0, 1.0])
    disp.plot(cmap=pyplot.cm.gray_r)
    assert_allclose(disp.text_[0, 1].get_color(), [0.0, 0.0, 0.0, 1.0])
    assert_allclose(disp.text_[1, 0].get_color(), [0.0, 0.0, 0.0, 1.0])
    assert_allclose(disp.text_[0, 0].get_color(), [1.0, 1.0, 1.0, 1.0])
    assert_allclose(disp.text_[1, 1].get_color(), [1.0, 1.0, 1.0, 1.0])
    cm = np.array([[19, 34], [32, 58]])
    disp = ConfusionMatrixDisplay(cm, display_labels=[0, 1])
    disp.plot(cmap=pyplot.cm.Blues)
    min_color = pyplot.cm.Blues(0)
    max_color = pyplot.cm.Blues(255)
    assert_allclose(disp.text_[0, 0].get_color(), max_color)
    assert_allclose(disp.text_[0, 1].get_color(), max_color)
    assert_allclose(disp.text_[1, 0].get_color(), max_color)
    assert_allclose(disp.text_[1, 1].get_color(), min_color)
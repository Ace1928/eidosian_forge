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
def test_confusion_matrix_text_kw(pyplot):
    """Check that text_kw is passed to the text call."""
    font_size = 15.0
    X, y = make_classification(random_state=0)
    classifier = SVC().fit(X, y)
    disp = ConfusionMatrixDisplay.from_estimator(classifier, X, y, text_kw={'fontsize': font_size})
    for text in disp.text_.reshape(-1):
        assert text.get_fontsize() == font_size
    new_font_size = 20.0
    disp.plot(text_kw={'fontsize': new_font_size})
    for text in disp.text_.reshape(-1):
        assert text.get_fontsize() == new_font_size
    y_pred = classifier.predict(X)
    disp = ConfusionMatrixDisplay.from_predictions(y, y_pred, text_kw={'fontsize': font_size})
    for text in disp.text_.reshape(-1):
        assert text.get_fontsize() == font_size
import numpy as np
import pytest
from numpy.testing import assert_allclose
from sklearn.base import BaseEstimator, clone
from sklearn.calibration import (
from sklearn.datasets import load_iris, make_blobs, make_classification
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
from sklearn.exceptions import NotFittedError
from sklearn.feature_extraction import DictVectorizer
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import (
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils._mocking import CheckingClassifier
from sklearn.utils._testing import (
from sklearn.utils.extmath import softmax
from sklearn.utils.fixes import CSR_CONTAINERS
@pytest.mark.parametrize('dtype_y_str', [str, object])
def test_calibration_curve_pos_label(dtype_y_str):
    """Check the behaviour when passing explicitly `pos_label`."""
    y_true = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1])
    classes = np.array(['spam', 'egg'], dtype=dtype_y_str)
    y_true_str = classes[y_true]
    y_pred = np.array([0.1, 0.2, 0.3, 0.4, 0.65, 0.7, 0.8, 0.9, 1.0])
    prob_true, _ = calibration_curve(y_true, y_pred, n_bins=4)
    assert_allclose(prob_true, [0, 0.5, 1, 1])
    prob_true, _ = calibration_curve(y_true_str, y_pred, n_bins=4, pos_label='egg')
    assert_allclose(prob_true, [0, 0.5, 1, 1])
    prob_true, _ = calibration_curve(y_true, 1 - y_pred, n_bins=4, pos_label=0)
    assert_allclose(prob_true, [0, 0, 0.5, 1])
    prob_true, _ = calibration_curve(y_true_str, 1 - y_pred, n_bins=4, pos_label='spam')
    assert_allclose(prob_true, [0, 0, 0.5, 1])
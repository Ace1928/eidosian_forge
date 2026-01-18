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
def test_calibration_curve_pos_label_error_str(dtype_y_str):
    """Check error message when a `pos_label` is not specified with `str` targets."""
    rng = np.random.RandomState(42)
    y1 = np.array(['spam'] * 3 + ['eggs'] * 2, dtype=dtype_y_str)
    y2 = rng.randint(0, 2, size=y1.size)
    err_msg = "y_true takes value in {'eggs', 'spam'} and pos_label is not specified: either make y_true take value in {0, 1} or {-1, 1} or pass pos_label explicitly"
    with pytest.raises(ValueError, match=err_msg):
        calibration_curve(y1, y2)
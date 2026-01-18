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
@pytest.mark.parametrize('X', [np.random.RandomState(42).randn(15, 5, 2), np.random.RandomState(42).randn(15, 5, 2, 6)])
def test_calibration_accepts_ndarray(X):
    """Test that calibration accepts n-dimensional arrays as input"""
    y = [1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0]

    class MockTensorClassifier(BaseEstimator):
        """A toy estimator that accepts tensor inputs"""
        _estimator_type = 'classifier'

        def fit(self, X, y):
            self.classes_ = np.unique(y)
            return self

        def decision_function(self, X):
            return X.reshape(X.shape[0], -1).sum(axis=1)
    calibrated_clf = CalibratedClassifierCV(MockTensorClassifier())
    calibrated_clf.fit(X, y)
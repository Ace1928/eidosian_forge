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
def test_calibration_without_sample_weight_estimator(data):
    """Check that even if the estimator doesn't support
    sample_weight, fitting with sample_weight still works.

    There should be a warning, since the sample_weight is not passed
    on to the estimator.
    """
    X, y = data
    sample_weight = np.ones_like(y)

    class ClfWithoutSampleWeight(CheckingClassifier):

        def fit(self, X, y, **fit_params):
            assert 'sample_weight' not in fit_params
            return super().fit(X, y, **fit_params)
    clf = ClfWithoutSampleWeight()
    pc_clf = CalibratedClassifierCV(clf)
    with pytest.warns(UserWarning):
        pc_clf.fit(X, y, sample_weight=sample_weight)
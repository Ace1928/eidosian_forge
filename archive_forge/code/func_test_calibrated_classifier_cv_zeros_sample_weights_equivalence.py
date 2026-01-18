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
@pytest.mark.parametrize('method', ['sigmoid', 'isotonic'])
@pytest.mark.parametrize('ensemble', [True, False])
def test_calibrated_classifier_cv_zeros_sample_weights_equivalence(method, ensemble):
    """Check that passing removing some sample from the dataset `X` is
    equivalent to passing a `sample_weight` with a factor 0."""
    X, y = load_iris(return_X_y=True)
    X = StandardScaler().fit_transform(X)
    X = np.vstack((X[:40], X[50:90]))
    y = np.hstack((y[:40], y[50:90]))
    sample_weight = np.zeros_like(y)
    sample_weight[::2] = 1
    estimator = LogisticRegression()
    calibrated_clf_without_weights = CalibratedClassifierCV(estimator, method=method, ensemble=ensemble, cv=2)
    calibrated_clf_with_weights = clone(calibrated_clf_without_weights)
    calibrated_clf_with_weights.fit(X, y, sample_weight=sample_weight)
    calibrated_clf_without_weights.fit(X[::2], y[::2])
    for est_with_weights, est_without_weights in zip(calibrated_clf_with_weights.calibrated_classifiers_, calibrated_clf_without_weights.calibrated_classifiers_):
        assert_allclose(est_with_weights.estimator.coef_, est_without_weights.estimator.coef_)
    y_pred_with_weights = calibrated_clf_with_weights.predict_proba(X)
    y_pred_without_weights = calibrated_clf_without_weights.predict_proba(X)
    assert_allclose(y_pred_with_weights, y_pred_without_weights)
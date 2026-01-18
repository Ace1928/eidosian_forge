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
def test_calibrated_classifier_cv_works_with_large_confidence_scores(global_random_seed):
    """Test that :class:`CalibratedClassifierCV` works with large confidence
    scores when using the `sigmoid` method, particularly with the
    :class:`SGDClassifier`.

    Non-regression test for issue #26766.
    """
    prob = 0.67
    n = 1000
    random_noise = np.random.default_rng(global_random_seed).normal(size=n)
    y = np.array([1] * int(n * prob) + [0] * (n - int(n * prob)))
    X = 100000.0 * y.reshape((-1, 1)) + random_noise
    cv = check_cv(cv=None, y=y, classifier=True)
    indices = cv.split(X, y)
    for train, test in indices:
        X_train, y_train = (X[train], y[train])
        X_test = X[test]
        sgd_clf = SGDClassifier(loss='squared_hinge', random_state=global_random_seed)
        sgd_clf.fit(X_train, y_train)
        predictions = sgd_clf.decision_function(X_test)
        assert (predictions > 10000.0).any()
    clf_sigmoid = CalibratedClassifierCV(SGDClassifier(loss='squared_hinge', random_state=global_random_seed), method='sigmoid')
    score_sigmoid = cross_val_score(clf_sigmoid, X, y, scoring='roc_auc')
    clf_isotonic = CalibratedClassifierCV(SGDClassifier(loss='squared_hinge', random_state=global_random_seed), method='isotonic')
    score_isotonic = cross_val_score(clf_isotonic, X, y, scoring='roc_auc')
    assert_allclose(score_sigmoid, score_isotonic)
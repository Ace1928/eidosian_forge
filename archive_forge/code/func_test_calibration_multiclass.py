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
@pytest.mark.parametrize('seed', range(2))
def test_calibration_multiclass(method, ensemble, seed):

    def multiclass_brier(y_true, proba_pred, n_classes):
        Y_onehot = np.eye(n_classes)[y_true]
        return np.sum((Y_onehot - proba_pred) ** 2) / Y_onehot.shape[0]
    clf = LinearSVC(dual='auto', random_state=7)
    X, y = make_blobs(n_samples=500, n_features=100, random_state=seed, centers=10, cluster_std=15.0)
    y[y > 2] = 2
    n_classes = np.unique(y).shape[0]
    X_train, y_train = (X[::2], y[::2])
    X_test, y_test = (X[1::2], y[1::2])
    clf.fit(X_train, y_train)
    cal_clf = CalibratedClassifierCV(clf, method=method, cv=5, ensemble=ensemble)
    cal_clf.fit(X_train, y_train)
    probas = cal_clf.predict_proba(X_test)
    assert_allclose(np.sum(probas, axis=1), np.ones(len(X_test)))
    assert 0.65 < clf.score(X_test, y_test) < 0.95
    assert cal_clf.score(X_test, y_test) > 0.95 * clf.score(X_test, y_test)
    uncalibrated_brier = multiclass_brier(y_test, softmax(clf.decision_function(X_test)), n_classes=n_classes)
    calibrated_brier = multiclass_brier(y_test, probas, n_classes=n_classes)
    assert calibrated_brier < 1.1 * uncalibrated_brier
    clf = RandomForestClassifier(n_estimators=30, random_state=42)
    clf.fit(X_train, y_train)
    clf_probs = clf.predict_proba(X_test)
    uncalibrated_brier = multiclass_brier(y_test, clf_probs, n_classes=n_classes)
    cal_clf = CalibratedClassifierCV(clf, method=method, cv=5, ensemble=ensemble)
    cal_clf.fit(X_train, y_train)
    cal_clf_probs = cal_clf.predict_proba(X_test)
    calibrated_brier = multiclass_brier(y_test, cal_clf_probs, n_classes=n_classes)
    assert calibrated_brier < 1.1 * uncalibrated_brier
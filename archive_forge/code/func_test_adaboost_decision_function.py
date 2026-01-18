import re
import numpy as np
import pytest
from sklearn import datasets
from sklearn.base import BaseEstimator, clone
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.ensemble._weight_boosting import _samme_proba
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils import shuffle
from sklearn.utils._mocking import NoSampleWeightWrapper
from sklearn.utils._testing import (
from sklearn.utils.fixes import (
@pytest.mark.filterwarnings('ignore:The SAMME.R algorithm')
@pytest.mark.parametrize('algorithm', ['SAMME', 'SAMME.R'])
def test_adaboost_decision_function(algorithm, global_random_seed):
    """Check that the decision function respects the symmetric constraint for weak
    learners.

    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/issues/26520
    """
    n_classes = 3
    X, y = datasets.make_classification(n_classes=n_classes, n_clusters_per_class=1, random_state=global_random_seed)
    clf = AdaBoostClassifier(n_estimators=1, random_state=global_random_seed, algorithm=algorithm).fit(X, y)
    y_score = clf.decision_function(X)
    assert_allclose(y_score.sum(axis=1), 0, atol=1e-08)
    if algorithm == 'SAMME':
        assert set(np.unique(y_score)) == {1, -1 / (n_classes - 1)}
    for y_score in clf.staged_decision_function(X):
        assert_allclose(y_score.sum(axis=1), 0, atol=1e-08)
        if algorithm == 'SAMME':
            assert set(np.unique(y_score)) == {1, -1 / (n_classes - 1)}
    clf.set_params(n_estimators=5).fit(X, y)
    y_score = clf.decision_function(X)
    assert_allclose(y_score.sum(axis=1), 0, atol=1e-08)
    for y_score in clf.staged_decision_function(X):
        assert_allclose(y_score.sum(axis=1), 0, atol=1e-08)
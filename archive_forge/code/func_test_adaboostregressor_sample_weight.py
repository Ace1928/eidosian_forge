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
def test_adaboostregressor_sample_weight():
    rng = np.random.RandomState(42)
    X = np.linspace(0, 100, num=1000)
    y = 0.8 * X + 0.2 + rng.rand(X.shape[0]) * 0.0001
    X = X.reshape(-1, 1)
    X[-1] *= 10
    y[-1] = 10000
    regr_no_outlier = AdaBoostRegressor(estimator=LinearRegression(), n_estimators=1, random_state=0)
    regr_with_weight = clone(regr_no_outlier)
    regr_with_outlier = clone(regr_no_outlier)
    regr_with_outlier.fit(X, y)
    regr_no_outlier.fit(X[:-1], y[:-1])
    sample_weight = np.ones_like(y)
    sample_weight[-1] = 0
    regr_with_weight.fit(X, y, sample_weight=sample_weight)
    score_with_outlier = regr_with_outlier.score(X[:-1], y[:-1])
    score_no_outlier = regr_no_outlier.score(X[:-1], y[:-1])
    score_with_weight = regr_with_weight.score(X[:-1], y[:-1])
    assert score_with_outlier < score_no_outlier
    assert score_with_outlier < score_with_weight
    assert score_no_outlier == pytest.approx(score_with_weight)
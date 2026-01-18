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
def test_multidimensional_X():
    """
    Check that the AdaBoost estimators can work with n-dimensional
    data matrix
    """
    rng = np.random.RandomState(0)
    X = rng.randn(51, 3, 3)
    yc = rng.choice([0, 1], 51)
    yr = rng.randn(51)
    boost = AdaBoostClassifier(DummyClassifier(strategy='most_frequent'), algorithm='SAMME')
    boost.fit(X, yc)
    boost.predict(X)
    boost.predict_proba(X)
    boost = AdaBoostRegressor(DummyRegressor())
    boost.fit(X, yr)
    boost.predict(X)
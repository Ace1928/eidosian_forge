from itertools import cycle, product
import joblib
import numpy as np
import pytest
from sklearn.base import BaseEstimator
from sklearn.datasets import load_diabetes, load_iris, make_hastie_10_2
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.model_selection import GridSearchCV, ParameterGrid, train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, scale
from sklearn.random_projection import SparseRandomProjection
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils import check_random_state
from sklearn.utils._testing import assert_array_almost_equal, assert_array_equal
from sklearn.utils.fixes import CSC_CONTAINERS, CSR_CONTAINERS
def test_bagging_get_estimators_indices():
    rng = np.random.RandomState(0)
    X = rng.randn(13, 4)
    y = np.arange(13)

    class MyEstimator(DecisionTreeRegressor):
        """An estimator which stores y indices information at fit."""

        def fit(self, X, y):
            self._sample_indices = y
    clf = BaggingRegressor(estimator=MyEstimator(), n_estimators=1, random_state=0)
    clf.fit(X, y)
    assert_array_equal(clf.estimators_[0]._sample_indices, clf.estimators_samples_[0])
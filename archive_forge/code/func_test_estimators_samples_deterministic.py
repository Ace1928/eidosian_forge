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
def test_estimators_samples_deterministic():
    iris = load_iris()
    X, y = (iris.data, iris.target)
    base_pipeline = make_pipeline(SparseRandomProjection(n_components=2), LogisticRegression())
    clf = BaggingClassifier(estimator=base_pipeline, max_samples=0.5, random_state=0)
    clf.fit(X, y)
    pipeline_estimator_coef = clf.estimators_[0].steps[-1][1].coef_.copy()
    estimator = clf.estimators_[0]
    estimator_sample = clf.estimators_samples_[0]
    estimator_feature = clf.estimators_features_[0]
    X_train = X[estimator_sample][:, estimator_feature]
    y_train = y[estimator_sample]
    estimator.fit(X_train, y_train)
    assert_array_equal(estimator.steps[-1][1].coef_, pipeline_estimator_coef)
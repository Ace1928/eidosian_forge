import re
import warnings
from unittest.mock import Mock
import numpy as np
import pytest
from sklearn import datasets
from sklearn.base import BaseEstimator
from sklearn.cross_decomposition import CCA, PLSCanonical, PLSRegression
from sklearn.datasets import make_friedman1
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.exceptions import NotFittedError
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import (
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
from sklearn.utils._testing import (
@pytest.mark.parametrize('estimator', [Lasso(alpha=0.1, random_state=42), LassoCV(random_state=42), ElasticNet(l1_ratio=1, random_state=42), ElasticNetCV(l1_ratio=[1], random_state=42)])
def test_coef_default_threshold(estimator):
    X, y = datasets.make_classification(n_samples=100, n_features=10, n_informative=3, n_redundant=0, n_repeated=0, shuffle=False, random_state=0)
    transformer = SelectFromModel(estimator=estimator)
    transformer.fit(X, y)
    X_new = transformer.transform(X)
    mask = np.abs(transformer.estimator_.coef_) > 1e-05
    assert_array_almost_equal(X_new, X[:, mask])
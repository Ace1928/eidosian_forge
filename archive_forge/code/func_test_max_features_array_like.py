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
@pytest.mark.parametrize('max_features', [lambda X: round(len(X[0]) / 2), 2])
def test_max_features_array_like(max_features):
    X = [[0.87, -1.34, 0.31], [-2.79, -0.02, -0.85], [-1.34, -0.48, -2.55], [1.92, 1.48, 0.65]]
    y = [0, 1, 0, 1]
    clf = RandomForestClassifier(n_estimators=5, random_state=0)
    transformer = SelectFromModel(estimator=clf, max_features=max_features, threshold=-np.inf)
    X_trans = transformer.fit_transform(X, y)
    assert X_trans.shape[1] == transformer.max_features_
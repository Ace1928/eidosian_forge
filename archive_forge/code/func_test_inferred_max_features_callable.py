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
@pytest.mark.parametrize('max_features', [lambda X: 1, lambda X: X.shape[1], lambda X: min(X.shape[1], 10000)])
def test_inferred_max_features_callable(max_features):
    """Check max_features_ and output shape for callable max_features."""
    clf = RandomForestClassifier(n_estimators=5, random_state=0)
    transformer = SelectFromModel(estimator=clf, max_features=max_features, threshold=-np.inf)
    X_trans = transformer.fit_transform(data, y)
    assert transformer.max_features_ == max_features(data)
    assert X_trans.shape[1] == transformer.max_features_
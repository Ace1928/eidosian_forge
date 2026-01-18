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
@pytest.mark.parametrize('max_features', [lambda X: min(X.shape[1], 10000), lambda X: X.shape[1], lambda X: 1])
def test_max_features_callable_data(max_features):
    """Tests that the callable passed to `fit` is called on X."""
    clf = RandomForestClassifier(n_estimators=50, random_state=0)
    m = Mock(side_effect=max_features)
    transformer = SelectFromModel(estimator=clf, max_features=m, threshold=-np.inf)
    transformer.fit_transform(data, y)
    m.assert_called_with(data)
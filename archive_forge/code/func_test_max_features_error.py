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
@pytest.mark.parametrize('max_features, err_type, err_msg', [(data.shape[1] + 1, ValueError, 'max_features =='), (lambda X: 1.5, TypeError, 'max_features must be an instance of int, not float.'), (lambda X: data.shape[1] + 1, ValueError, 'max_features =='), (lambda X: -1, ValueError, 'max_features ==')])
def test_max_features_error(max_features, err_type, err_msg):
    err_msg = re.escape(err_msg)
    clf = RandomForestClassifier(n_estimators=5, random_state=0)
    transformer = SelectFromModel(estimator=clf, max_features=max_features, threshold=-np.inf)
    with pytest.raises(err_type, match=err_msg):
        transformer.fit(data, y)
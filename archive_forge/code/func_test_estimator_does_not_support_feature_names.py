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
def test_estimator_does_not_support_feature_names():
    """SelectFromModel works with estimators that do not support feature_names_in_.

    Non-regression test for #21949.
    """
    pytest.importorskip('pandas')
    X, y = datasets.load_iris(as_frame=True, return_X_y=True)
    all_feature_names = set(X.columns)

    def importance_getter(estimator):
        return np.arange(X.shape[1])
    selector = SelectFromModel(MinimalClassifier(), importance_getter=importance_getter).fit(X, y)
    assert_array_equal(selector.feature_names_in_, X.columns)
    feature_names_out = set(selector.get_feature_names_out())
    assert feature_names_out < all_feature_names
    with warnings.catch_warnings():
        warnings.simplefilter('error', UserWarning)
        selector.transform(X.iloc[1:3])
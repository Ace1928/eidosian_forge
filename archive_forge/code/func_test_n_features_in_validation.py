import pickle
import re
import warnings
import numpy as np
import pytest
import scipy.sparse as sp
from numpy.testing import assert_allclose
import sklearn
from sklearn import config_context, datasets
from sklearn.base import (
from sklearn.decomposition import PCA
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils._mocking import MockDataFrame
from sklearn.utils._set_output import _get_output_config
from sklearn.utils._testing import (
def test_n_features_in_validation():
    """Check that `_check_n_features` validates data when reset=False"""
    est = MyEstimator()
    X_train = [[1, 2, 3], [4, 5, 6]]
    est._check_n_features(X_train, reset=True)
    assert est.n_features_in_ == 3
    msg = 'X does not contain any features, but MyEstimator is expecting 3 features'
    with pytest.raises(ValueError, match=msg):
        est._check_n_features('invalid X', reset=False)
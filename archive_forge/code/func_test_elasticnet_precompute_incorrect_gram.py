import warnings
from copy import deepcopy
import joblib
import numpy as np
import pytest
from scipy import interpolate, sparse
from sklearn.base import clone, is_classifier
from sklearn.datasets import load_diabetes, make_regression
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import (
from sklearn.linear_model._coordinate_descent import _set_order
from sklearn.model_selection import (
from sklearn.model_selection._split import GroupsConsumerMixin
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_array
from sklearn.utils._testing import (
from sklearn.utils.fixes import COO_CONTAINERS, CSC_CONTAINERS, CSR_CONTAINERS
def test_elasticnet_precompute_incorrect_gram():
    X, y, _, _ = build_dataset()
    rng = np.random.RandomState(0)
    X_centered = X - np.average(X, axis=0)
    garbage = rng.standard_normal(X.shape)
    precompute = np.dot(garbage.T, garbage)
    clf = ElasticNet(alpha=0.01, precompute=precompute)
    msg = 'Gram matrix.*did not pass validation.*'
    with pytest.raises(ValueError, match=msg):
        clf.fit(X_centered, y)
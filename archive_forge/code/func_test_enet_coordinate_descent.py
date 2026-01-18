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
@pytest.mark.parametrize('klass, n_classes, kwargs', [(Lasso, 1, dict(precompute=True)), (Lasso, 1, dict(precompute=False)), (MultiTaskLasso, 2, dict()), (MultiTaskLasso, 2, dict())])
def test_enet_coordinate_descent(klass, n_classes, kwargs):
    """Test that a warning is issued if model does not converge"""
    clf = klass(max_iter=2, **kwargs)
    n_samples = 5
    n_features = 2
    X = np.ones((n_samples, n_features)) * 1e+50
    y = np.ones((n_samples, n_classes))
    if klass == Lasso:
        y = y.ravel()
    warning_message = 'Objective did not converge. You might want to increase the number of iterations.'
    with pytest.warns(ConvergenceWarning, match=warning_message):
        clf.fit(X, y)
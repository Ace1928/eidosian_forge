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
def test_enet_path_positive():
    X, Y, _, _ = build_dataset(n_samples=50, n_features=50, n_targets=2)
    for path in [enet_path, lasso_path]:
        pos_path_coef = path(X, Y[:, 0], positive=True)[1]
        assert np.all(pos_path_coef >= 0)
    for path in [enet_path, lasso_path]:
        with pytest.raises(ValueError):
            path(X, Y, positive=True)
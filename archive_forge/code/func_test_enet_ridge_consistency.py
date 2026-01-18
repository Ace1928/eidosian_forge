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
@pytest.mark.parametrize('ridge_alpha', [0.1, 1.0, 1000000.0])
def test_enet_ridge_consistency(ridge_alpha):
    rng = np.random.RandomState(42)
    n_samples = 300
    X, y = make_regression(n_samples=n_samples, n_features=100, effective_rank=10, n_informative=50, random_state=rng)
    sw = rng.uniform(low=0.01, high=10, size=X.shape[0])
    alpha = 1.0
    common_params = dict(tol=1e-12)
    ridge = Ridge(alpha=alpha, **common_params).fit(X, y, sample_weight=sw)
    alpha_enet = alpha / sw.sum()
    enet = ElasticNet(alpha=alpha_enet, l1_ratio=0, **common_params).fit(X, y, sample_weight=sw)
    assert_allclose(ridge.coef_, enet.coef_)
    assert_allclose(ridge.intercept_, enet.intercept_)
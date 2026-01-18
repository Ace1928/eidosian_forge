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
@pytest.mark.parametrize('alphas, err_type, err_msg', [((1, -1, -100), ValueError, 'alphas\\[1\\] == -1, must be >= 0.0.'), ((-0.1, -1.0, -10.0), ValueError, 'alphas\\[0\\] == -0.1, must be >= 0.0.'), ((1, 1.0, '1'), TypeError, 'alphas\\[2\\] must be an instance of float, not str')])
def test_lassocv_alphas_validation(alphas, err_type, err_msg):
    """Check the `alphas` validation in LassoCV."""
    n_samples, n_features = (5, 5)
    rng = np.random.RandomState(0)
    X = rng.randn(n_samples, n_features)
    y = rng.randint(0, 2, n_samples)
    lassocv = LassoCV(alphas=alphas)
    with pytest.raises(err_type, match=err_msg):
        lassocv.fit(X, y)
import sys
from io import StringIO
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import linalg
from sklearn import datasets
from sklearn.covariance import (
from sklearn.datasets import make_sparse_spd_matrix
from sklearn.utils import check_random_state
from sklearn.utils._testing import (
@pytest.mark.parametrize('alphas_container_type', ['list', 'tuple', 'array'])
def test_graphical_lasso_cv_alphas_iterable(alphas_container_type):
    """Check that we can pass an array-like to `alphas`.

    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/issues/22489
    """
    true_cov = np.array([[0.8, 0.0, 0.2, 0.0], [0.0, 0.4, 0.0, 0.0], [0.2, 0.0, 0.3, 0.1], [0.0, 0.0, 0.1, 0.7]])
    rng = np.random.RandomState(0)
    X = rng.multivariate_normal(mean=[0, 0, 0, 0], cov=true_cov, size=200)
    alphas = _convert_container([0.02, 0.03], alphas_container_type)
    GraphicalLassoCV(alphas=alphas, tol=0.1, n_jobs=1).fit(X)
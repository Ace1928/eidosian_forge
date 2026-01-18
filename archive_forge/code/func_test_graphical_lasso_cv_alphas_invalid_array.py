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
@pytest.mark.parametrize('alphas,err_type,err_msg', [([-0.02, 0.03], ValueError, 'must be > 0'), ([0, 0.03], ValueError, 'must be > 0'), (['not_number', 0.03], TypeError, 'must be an instance of float')])
def test_graphical_lasso_cv_alphas_invalid_array(alphas, err_type, err_msg):
    """Check that if an array-like containing a value
    outside of (0, inf] is passed to `alphas`, a ValueError is raised.
    Check if a string is passed, a TypeError is raised.
    """
    true_cov = np.array([[0.8, 0.0, 0.2, 0.0], [0.0, 0.4, 0.0, 0.0], [0.2, 0.0, 0.3, 0.1], [0.0, 0.0, 0.1, 0.7]])
    rng = np.random.RandomState(0)
    X = rng.multivariate_normal(mean=[0, 0, 0, 0], cov=true_cov, size=200)
    with pytest.raises(err_type, match=err_msg):
        GraphicalLassoCV(alphas=alphas, tol=0.1, n_jobs=1).fit(X)
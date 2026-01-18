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
@pytest.mark.parametrize('mode', ['cd', 'lars'])
def test_graphical_lasso_n_iter(mode):
    X, _ = datasets.make_classification(n_samples=5000, n_features=20, random_state=0)
    emp_cov = empirical_covariance(X)
    _, _, n_iter = graphical_lasso(emp_cov, 0.2, mode=mode, max_iter=2, return_n_iter=True)
    assert n_iter == 2
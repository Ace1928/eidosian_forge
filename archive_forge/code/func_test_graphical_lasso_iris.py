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
def test_graphical_lasso_iris():
    cov_R = np.array([[0.68112222, 0.0, 0.26582, 0.02464314], [0.0, 0.1887129, 0.0, 0.0], [0.26582, 0.0, 3.095503, 0.286972], [0.02464314, 0.0, 0.286972, 0.57713289]])
    icov_R = np.array([[1.5190747, 0.0, -0.1304475, 0.0], [0.0, 5.299055, 0.0, 0.0], [-0.1304475, 0.0, 0.3498624, -0.1683946], [0.0, 0.0, -0.1683946, 1.8164353]])
    X = datasets.load_iris().data
    emp_cov = empirical_covariance(X)
    for method in ('cd', 'lars'):
        cov, icov = graphical_lasso(emp_cov, alpha=1.0, return_costs=False, mode=method)
        assert_array_almost_equal(cov, cov_R)
        assert_array_almost_equal(icov, icov_R)
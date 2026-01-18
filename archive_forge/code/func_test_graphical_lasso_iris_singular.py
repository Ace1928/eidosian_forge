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
def test_graphical_lasso_iris_singular():
    indices = np.arange(10, 13)
    cov_R = np.array([[0.08, 0.056666662595, 0.00229729713223, 0.00153153142149], [0.056666662595, 0.082222222222, 0.00333333333333, 0.00222222222222], [0.002297297132, 0.003333333333, 0.00666666666667, 9.009009009e-05], [0.001531531421, 0.002222222222, 9.009009009e-05, 0.00222222222222]])
    icov_R = np.array([[24.42244057, -16.831679593, 0.0, 0.0], [-16.83168201, 24.351841681, -6.206896552, -12.5], [0.0, -6.206896171, 153.103448276, 0.0], [0.0, -12.499999143, 0.0, 462.5]])
    X = datasets.load_iris().data[indices, :]
    emp_cov = empirical_covariance(X)
    for method in ('cd', 'lars'):
        cov, icov = graphical_lasso(emp_cov, alpha=0.01, return_costs=False, mode=method)
        assert_array_almost_equal(cov, cov_R, decimal=5)
        assert_array_almost_equal(icov, icov_R, decimal=5)
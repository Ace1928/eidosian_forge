import re
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal
from scipy.optimize import check_grad
from sklearn import clone
from sklearn.datasets import load_iris, make_blobs, make_classification
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import check_random_state
def test_finite_differences(global_random_seed):
    """Test gradient of loss function

    Assert that the gradient is almost equal to its finite differences
    approximation.
    """
    rng = np.random.RandomState(global_random_seed)
    X, y = make_classification(random_state=global_random_seed)
    M = rng.randn(rng.randint(1, X.shape[1] + 1), X.shape[1])
    nca = NeighborhoodComponentsAnalysis()
    nca.n_iter_ = 0
    mask = y[:, np.newaxis] == y[np.newaxis, :]

    def fun(M):
        return nca._loss_grad_lbfgs(M, X, mask)[0]

    def grad(M):
        return nca._loss_grad_lbfgs(M, X, mask)[1]
    diff = check_grad(fun, grad, M.ravel())
    assert diff == pytest.approx(0.0, abs=0.0001)
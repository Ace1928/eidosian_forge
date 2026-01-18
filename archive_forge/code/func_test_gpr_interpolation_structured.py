import re
import sys
import warnings
import numpy as np
import pytest
from scipy.optimize import approx_fprime
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
from sklearn.gaussian_process.kernels import (
from sklearn.gaussian_process.tests._mini_sequence_kernel import MiniSeqKernel
from sklearn.utils._testing import (
def test_gpr_interpolation_structured():
    kernel = MiniSeqKernel(baseline_similarity_bounds='fixed')
    X = ['A', 'B', 'C']
    y = np.array([1, 2, 3])
    gpr = GaussianProcessRegressor(kernel=kernel).fit(X, y)
    y_pred, y_cov = gpr.predict(X, return_cov=True)
    assert_almost_equal(kernel(X, eval_gradient=True)[1].ravel(), (1 - np.eye(len(X))).ravel())
    assert_almost_equal(y_pred, y)
    assert_almost_equal(np.diag(y_cov), 0.0)
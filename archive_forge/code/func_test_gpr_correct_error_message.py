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
def test_gpr_correct_error_message():
    X = np.arange(12).reshape(6, -1)
    y = np.ones(6)
    kernel = DotProduct()
    gpr = GaussianProcessRegressor(kernel=kernel, alpha=0.0)
    message = "The kernel, %s, is not returning a positive definite matrix. Try gradually increasing the 'alpha' parameter of your GaussianProcessRegressor estimator." % kernel
    with pytest.raises(np.linalg.LinAlgError, match=re.escape(message)):
        gpr.fit(X, y)
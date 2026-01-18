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
def test_no_optimizer():
    kernel = RBF(1.0)
    gpr = GaussianProcessRegressor(kernel=kernel, optimizer=None).fit(X, y)
    assert np.exp(gpr.kernel_.theta) == 1.0
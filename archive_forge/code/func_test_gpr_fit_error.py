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
@pytest.mark.parametrize('params, TypeError, err_msg', [({'alpha': np.zeros(100)}, ValueError, 'alpha must be a scalar or an array with same number of entries as y'), ({'kernel': WhiteKernel(noise_level_bounds=(-np.inf, np.inf)), 'n_restarts_optimizer': 2}, ValueError, 'requires that all bounds are finite')])
def test_gpr_fit_error(params, TypeError, err_msg):
    """Check that expected error are raised during fit."""
    gpr = GaussianProcessRegressor(**params)
    with pytest.raises(TypeError, match=err_msg):
        gpr.fit(X, y)
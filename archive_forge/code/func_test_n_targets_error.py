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
def test_n_targets_error():
    """Check that an error is raised when the number of targets seen at fit is
    inconsistent with n_targets.
    """
    rng = np.random.RandomState(0)
    X = rng.randn(10, 3)
    y = rng.randn(10, 2)
    model = GaussianProcessRegressor(n_targets=1)
    with pytest.raises(ValueError, match='The number of targets seen in `y`'):
        model.fit(X, y)
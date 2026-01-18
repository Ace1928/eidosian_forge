import operator
import sys
import time
import warnings
from numbers import Integral, Real
import numpy as np
from scipy import linalg
from ..base import _fit_context
from ..exceptions import ConvergenceWarning
from ..linear_model import _cd_fast as cd_fast  # type: ignore
from ..linear_model import lars_path_gram
from ..model_selection import check_cv, cross_val_score
from ..utils._param_validation import Interval, StrOptions, validate_params
from ..utils.metadata_routing import _RoutingNotSupportedMixin
from ..utils.parallel import Parallel, delayed
from ..utils.validation import (
from . import EmpiricalCovariance, empirical_covariance, log_likelihood
def alpha_max(emp_cov):
    """Find the maximum alpha for which there are some non-zeros off-diagonal.

    Parameters
    ----------
    emp_cov : ndarray of shape (n_features, n_features)
        The sample covariance matrix.

    Notes
    -----
    This results from the bound for the all the Lasso that are solved
    in GraphicalLasso: each time, the row of cov corresponds to Xy. As the
    bound for alpha is given by `max(abs(Xy))`, the result follows.
    """
    A = np.copy(emp_cov)
    A.flat[::A.shape[0] + 1] = 0
    return np.max(np.abs(A))
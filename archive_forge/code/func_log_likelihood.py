import warnings
import numpy as np
from scipy import linalg
from .. import config_context
from ..base import BaseEstimator, _fit_context
from ..metrics.pairwise import pairwise_distances
from ..utils import check_array
from ..utils._param_validation import validate_params
from ..utils.extmath import fast_logdet
@validate_params({'emp_cov': [np.ndarray], 'precision': [np.ndarray]}, prefer_skip_nested_validation=True)
def log_likelihood(emp_cov, precision):
    """Compute the sample mean of the log_likelihood under a covariance model.

    Computes the empirical expected log-likelihood, allowing for universal
    comparison (beyond this software package), and accounts for normalization
    terms and scaling.

    Parameters
    ----------
    emp_cov : ndarray of shape (n_features, n_features)
        Maximum Likelihood Estimator of covariance.

    precision : ndarray of shape (n_features, n_features)
        The precision matrix of the covariance model to be tested.

    Returns
    -------
    log_likelihood_ : float
        Sample mean of the log-likelihood.
    """
    p = precision.shape[0]
    log_likelihood_ = -np.sum(emp_cov * precision) + fast_logdet(precision)
    log_likelihood_ -= p * np.log(2 * np.pi)
    log_likelihood_ /= 2.0
    return log_likelihood_